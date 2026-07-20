from pathlib import Path
import sentencepiece as spm

class ToolCallTokenizer:
    def __init__(self, model_path: str | Path):
        self.processor = spm.SentencePieceProcessor(model_file=str(model_path))

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> list[int]:
        ids = self.processor.encode(text, out_type=int)
        if add_bos:
            ids.insert(0, self.processor.bos_id())
        if add_eos:
            ids.append(self.processor.eos_id())
        return ids

    def decode(self, token_ids: list[int]) -> str:
        return self.processor.decode(token_ids)