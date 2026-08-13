#!/usr/bin/env python3
"""Fail fast unless the data generator is using its tested dependency set."""

from __future__ import annotations

from importlib.metadata import version


EXPECTED = {
    "datasets": "3.2.0",
    "huggingface-hub": "0.27.1",
    "fsspec": "2024.9.0",
    "pyarrow": "17.0.0",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "PyYAML": "6.0.2",
    "sentencepiece": "0.2.0",
    "tqdm": "4.67.1",
}


def main() -> None:
    actual = {package: version(package) for package in EXPECTED}
    mismatches = {
        package: {"expected": EXPECTED[package], "actual": actual[package]}
        for package in EXPECTED
        if actual[package] != EXPECTED[package]
    }
    if mismatches:
        raise RuntimeError(f"Pinned data runtime mismatch: {mismatches}")
    print(
        "Pinned data runtime: "
        + " ".join(f"{package}={actual[package]}" for package in EXPECTED)
    )


if __name__ == "__main__":
    main()
