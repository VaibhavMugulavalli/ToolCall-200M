from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sft_builder", PROJECT / "scripts" / "sft_builder.py"
)
assert SPEC and SPEC.loader
sft = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sft
SPEC.loader.exec_module(sft)


def make_tool(index: int, required: bool = True, value_type: str = "string") -> dict:
    return {
        "name": f"service_{index}.execute",
        "description": f"Execute operation {index} for a distinct service domain",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": value_type, "description": "Input value"},
                "mode": {"type": "string", "enum": ["safe", "fast"]},
            },
            "required": ["value"] if required else [],
            "additionalProperties": False,
        },
    }


class BuilderTests(unittest.TestCase):
    def test_config_totals(self) -> None:
        config = json.loads((PROJECT / "configs" / "sft_v1.json").read_text())
        self.assertEqual(sum(config["category_targets"].values()), 150_000)
        self.assertEqual(sum(config["split_targets"].values()), 150_000)
        self.assertEqual(sum(config["pilot_targets"].values()), 1_000)
        for category, total in config["category_targets"].items():
            split_total = sum(
                sft.category_split_target(total, split)
                for split in ("train", "validation", "test")
            )
            self.assertEqual(split_total, total, category)

    def test_xlam_style_tool_normalization(self) -> None:
        raw = {
            "name": "weather.forecast",
            "description": "Forecast weather",
            "parameters": {
                "city": {"type": "str", "required": True},
                "days": {"type": "int", "required": False},
            },
        }
        tool = sft.normalize_tool(raw)
        self.assertEqual(tool["parameters"]["required"], ["city"])
        self.assertEqual(tool["parameters"]["properties"]["city"]["type"], "string")
        self.assertEqual(tool["parameters"]["properties"]["days"]["type"], "integer")

    def test_actual_xlam_legacy_optionality_convention(self) -> None:
        raw = {
            "name": "calculate_return",
            "description": "Calculate an investment return",
            "parameters": {
                "initial_amount": {"type": "float", "description": "Deposit"},
                "years": {"type": "int", "description": "Years"},
                "currency": {
                    "type": "str, optional",
                    "description": "Currency code",
                },
                "rounding": {
                    "type": "int",
                    "description": "Decimal places",
                    "default": 2,
                },
            },
        }
        tool = sft.normalize_tool(raw)
        self.assertEqual(
            tool["parameters"]["required"],
            ["initial_amount", "years"],
        )
        self.assertEqual(
            tool["parameters"]["properties"]["currency"]["type"],
            "string",
        )

    def test_decision_invariants(self) -> None:
        tool = make_tool(1)
        sft.validate_target(
            {
                "decision": "call",
                "tool_name": tool["name"],
                "arguments": {"value": "x"},
                "missing_required_fields": [],
            },
            [tool],
        )
        sft.validate_target(
            {
                "decision": "ask_clarification",
                "tool_name": tool["name"],
                "arguments": {},
                "missing_required_fields": ["value"],
            },
            [tool],
        )
        sft.validate_target(
            {
                "decision": "no_call",
                "tool_name": None,
                "arguments": {},
                "missing_required_fields": [],
            },
            [tool],
        )
        with self.assertRaises(ValueError):
            sft.validate_target(
                {
                    "decision": "call",
                    "tool_name": tool["name"],
                    "arguments": {},
                    "missing_required_fields": [],
                    "confidence": 1.0,
                },
                [tool],
            )

    def test_nested_confidence_argument_is_not_a_contract_field(self) -> None:
        tool = {
            "name": "classifier.submit",
            "description": "Submit a classification result",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["label", "confidence"],
                "additionalProperties": False,
            },
        }
        target = {
            "decision": "call",
            "tool_name": tool["name"],
            "arguments": {"label": "positive", "confidence": 0.91},
            "missing_required_fields": [],
        }
        record = {
            "tools": [tool],
            "target": target,
            "target_text": sft.canonical_json(target),
        }
        sft.validate_rendered_target(record)
        self.assertNotIn("confidence", target)
        self.assertIn("confidence", target["arguments"])

    def test_rendered_target_must_equal_structured_target(self) -> None:
        tool = make_tool(1)
        target = {
            "decision": "call",
            "tool_name": tool["name"],
            "arguments": {"value": "x"},
            "missing_required_fields": [],
        }
        record = {
            "tools": [tool],
            "target": target,
            "target_text": sft.canonical_json({**target, "confidence": 0.9}),
        }
        with self.assertRaisesRegex(ValueError, "does not match"):
            sft.validate_rendered_target(record)

    def test_all_transformations(self) -> None:
        tools = [make_tool(index, value_type="integer") for index in range(30)]
        pools = {"train": tools, "validation": [], "test": []}
        selector = sft.ToolSelector(pools)
        seed = sft.Seed(
            source_id="1",
            source_position=1,
            query="Run operation one with value 7 in safe mode",
            tool=tools[0],
            arguments={"value": 7, "mode": "safe"},
            schema_family_id=sft.tool_signature(tools[0]),
            split="train",
        )
        for category in (
            "valid_call",
            "missing_required_fields",
            "no_applicable_tool",
            "confusable_tools",
            "schema_constraints",
            "injection_resistance",
        ):
            candidate = sft.candidate_for(category, seed, 1, selector, 6)
            self.assertIsNotNone(candidate, category)
            request, available_tools, target, _metadata = candidate
            self.assertTrue(request)
            sft.validate_target(target, available_tools)
            self.assertNotIn("confidence", target)
        no_call = sft.candidate_for("no_applicable_tool", seed, 1, selector, 6)
        self.assertNotIn(seed.tool["name"], [tool["name"] for tool in no_call[1]])

    def test_optional_field_is_promoted_for_clarification(self) -> None:
        tools = [make_tool(0, required=False), *[make_tool(index) for index in range(1, 20)]]
        selector = sft.ToolSelector({"train": tools, "validation": [], "test": []})
        seed = sft.Seed(
            source_id="optional",
            source_position=9,
            query="Run the optional operation with value alpha",
            tool=tools[0],
            arguments={"value": "alpha", "mode": "safe"},
            schema_family_id=sft.tool_signature(tools[0]),
            split="train",
        )
        candidate = sft.candidate_for(
            "missing_required_fields", seed, 0, selector, 6
        )
        self.assertIsNotNone(candidate)
        _request, available_tools, target, metadata = candidate
        selected = next(tool for tool in available_tools if tool["name"] == tools[0]["name"])
        missing = target["missing_required_fields"][0]
        self.assertIn(missing, selected["parameters"]["required"])
        self.assertNotIn(missing, target["arguments"])
        self.assertIn(metadata["schema_adaptation"], {
            "none", "promoted_optional_to_required"
        })
        sft.validate_target(target, available_tools)

    def test_distractors_have_unique_non_target_names(self) -> None:
        correct = make_tool(0)
        duplicate_name = make_tool(1)
        duplicate_name["name"] = correct["name"]
        tools = [correct, duplicate_name, *[make_tool(index) for index in range(2, 20)]]
        selector = sft.ToolSelector({"train": tools, "validation": [], "test": []})
        chosen = selector.choose("train", correct, 6, "unique-names", similar=True)
        names = [tool["name"] for tool in chosen]
        self.assertNotIn(correct["name"], names)
        self.assertEqual(len(names), len(set(names)))

    def test_v102_migration_preserves_completed_work(self) -> None:
        config = json.loads((PROJECT / "configs" / "sft_v1.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "work" / "train").mkdir(parents=True)
            (root / "state").mkdir(parents=True)
            valid = root / "work" / "train" / "valid_call.jsonl"
            valid.write_text('{"id":"preserve"}\n', encoding="utf-8")
            partial = root / "work" / "train" / "missing_required_fields.jsonl"
            partial.write_text('{"id":"old-a"}\n{"id":"old-b"}\n', encoding="utf-8")

            sft.migrate_v102(root, config)

            self.assertTrue(valid.is_file())
            self.assertFalse(partial.exists())
            archive = (
                root / "migrations" / "v1_0_2" /
                "train_missing_required_fields.pre_v1_0_2.jsonl"
            )
            self.assertEqual(sft.count_jsonl(archive), 2)
            self.assertTrue((root / "state" / "migration_v1_0_2.json").is_file())

    def test_clarification_generator_exceeds_one_row_per_seed(self) -> None:
        class FakeProcessor:
            def __init__(self, **_kwargs):
                pass

            def vocab_size(self):
                return 32_000

            def eos_id(self):
                return 2

            def encode(self, text, out_type=int):
                return list(range(min(500, max(1, len(text.split())))))

        fake_sentencepiece = types.SimpleNamespace(SentencePieceProcessor=FakeProcessor)
        tools = [make_tool(index, required=False) for index in range(240)]
        selector = sft.ToolSelector({"train": tools, "validation": [], "test": []})
        seeds = [
            sft.Seed(
                source_id=str(index),
                source_position=index,
                query=f"Run operation {index} using value value-{index}",
                tool=tools[index],
                arguments={"value": f"value-{index}", "mode": "safe"},
                schema_family_id=sft.tool_signature(tools[index]),
                split="train",
            )
            for index in range(len(tools))
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = root / "tokenizer.model"
            tokenizer.write_bytes(b"test")
            output = root / "missing.jsonl"
            with mock.patch.dict(sys.modules, {"sentencepiece": fake_sentencepiece}):
                factory = sft.RecordFactory(tokenizer, 2048, 2)
                accepted = sft.generate_records(
                    factory,
                    "missing_required_fields",
                    "train",
                    seeds,
                    selector,
                    360,
                    6,
                    12,
                    output,
                    append=False,
                )
        self.assertEqual(accepted, 360)

    def test_weighted_interleave_preserves_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            expected = set()
            for category, count in (("a", 7), ("b", 3), ("c", 1)):
                path = root / f"{category}.jsonl"
                paths[category] = path
                with path.open("w", encoding="utf-8") as handle:
                    for index in range(count):
                        record = {"id": f"{category}{index}"}
                        expected.add(record["id"])
                        handle.write(json.dumps(record) + "\n")
            actual = {row["id"] for row in sft.weighted_interleave(paths)}
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
