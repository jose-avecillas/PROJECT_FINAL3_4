from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

DEFAULT_ZS_MODEL = "joeddav/xlm-roberta-large-xnli"
DEFAULT_BANKING77_MODEL = "mrm8488/bert-mini-finetuned-banking77"

def build_zero_shot_pipeline(model_name: str = DEFAULT_ZS_MODEL):
    return pipeline("zero-shot-classification", model=model_name)

def infer_zeroshot(zs_pipe, text: str, candidate_labels: List[str], multi_label: bool = False) -> Dict[str, Any]:
    res = zs_pipe(text, candidate_labels=candidate_labels, multi_label=multi_label)
    labels = res["labels"]; scores = res["scores"]
    return {"top_label": labels[0], "top_score": float(scores[0]),
            "ranked": [{"label": l, "score": float(s)} for l, s in zip(labels, scores)]}

def build_banking77_pipeline(model_path_or_name: str = DEFAULT_BANKING77_MODEL):
    tok = AutoTokenizer.from_pretrained(model_path_or_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_path_or_name)
    return pipeline("text-classification", model=mdl, tokenizer=tok, top_k=None, truncation=True)

def infer_banking77(cls_pipe, text: str, topk: int = 5) -> Dict[str, Any]:
    out = cls_pipe(text)
    ranked = sorted(out, key=lambda d: d["score"], reverse=True)[:topk]
    return {"top_label": ranked[0]["label"], "top_score": float(ranked[0]["score"]),
            "ranked": [{"label": d["label"], "score": float(d["score"])} for d in ranked]}
