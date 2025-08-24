from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from src.nlp.intent import (
    build_zero_shot_pipeline, build_banking77_pipeline,
    infer_zeroshot, infer_banking77, DEFAULT_ZS_MODEL, DEFAULT_BANKING77_MODEL
)
from src.nlp.dialogue import MemoryBuffer, dialog_reply

app = FastAPI(title="Proyecto IA API")
_zs_pipe = None; _ft_pipe = None
def get_zs_pipe(model_name: str = DEFAULT_ZS_MODEL):
    global _zs_pipe
    if _zs_pipe is None:
        _zs_pipe = build_zero_shot_pipeline(model_name)
    return _zs_pipe
def get_ft_pipe(model_path_or_name: str = DEFAULT_BANKING77_MODEL):
    global _ft_pipe
    if _ft_pipe is None:
        _ft_pipe = build_banking77_pipeline(model_path_or_name)
    return _ft_pipe

MEM = MemoryBuffer(max_turns=6)

class IntentZSReq(BaseModel):
    text: str
    candidate_labels: List[str]
    multi_label: bool = False
    model_name: Optional[str] = None

class IntentFTReq(BaseModel):
    text: str
    model_path_or_name: Optional[str] = None

class DialogReq(BaseModel):
    user_text: str

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/intent/zeroshot")
def intent_zeroshot(req: IntentZSReq) -> Dict[str, Any]:
    pipe = get_zs_pipe(req.model_name or DEFAULT_ZS_MODEL)
    return infer_zeroshot(pipe, req.text, req.candidate_labels, req.multi_label)

@app.post("/intent/finetuned")
def intent_finetuned(req: IntentFTReq) -> Dict[str, Any]:
    pipe = get_ft_pipe(req.model_path_or_name or DEFAULT_BANKING77_MODEL)
    return infer_banking77(pipe, req.text)

@app.post("/dialog")
def dialog(req: DialogReq) -> Dict[str, Any]:
    reply = dialog_reply(MEM, req.user_text)
    return {"reply": reply}
