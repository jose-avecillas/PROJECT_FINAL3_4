from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

DIALOG_MODEL_NAME = "microsoft/DialoGPT-small"
_dlg_tok = None; _dlg_model = None

def _ensure_model():
    global _dlg_tok, _dlg_model
    if _dlg_tok is None or _dlg_model is None:
        _dlg_tok = AutoTokenizer.from_pretrained(DIALOG_MODEL_NAME)
        _dlg_model = AutoModelForCausalLM.from_pretrained(DIALOG_MODEL_NAME)
    return _dlg_tok, _dlg_model

class MemoryBuffer:
    def __init__(self, max_turns: int = 6):
        self.buffer: List[Tuple[str, str]] = []
        self.max_turns = max_turns
    def add(self, speaker: str, text: str):
        self.buffer.append((speaker, text))
        self.buffer = self.buffer[-self.max_turns:]
    def as_text(self) -> str:
        return "\n".join([f"{s}: {t}" for s, t in self.buffer])

def dialog_reply(memory: MemoryBuffer, user_text: str, max_new_tokens: int = 80) -> str:
    tok, model = _ensure_model()
    memory.add("User", user_text)
    prompt = memory.as_text() + "\nBot:"
    enc = tok.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        out_ids = model.generate(enc, max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id,
                                 do_sample=True, top_p=0.9, temperature=0.7)
    reply = tok.decode(out_ids[0], skip_special_tokens=True).split("Bot:")[-1].strip()
    memory.add("Bot", reply)
    return reply
