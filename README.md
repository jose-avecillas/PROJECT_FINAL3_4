# Proyecto IA – Chatbot (estructura base)


## Intent (Zero-shot y Fine-tuned)
- Zero-shot (multilingüe):
```bash
curl -X POST http://127.0.0.1:8000/intent/zeroshot       -H "Content-Type: application/json"       -d '{"text":"Quiero saber mi saldo", "candidate_labels":["check_balance","transfer","lost_card"]}'
```
- Fine-tuned (Banking77):
```bash
curl -X POST http://127.0.0.1:8000/intent/finetuned       -H "Content-Type: application/json"       -d '{"text":"I lost my card", "model_path_or_name":"./modelo_banking77"}'
```

## Diálogo (DialoGPT)
```bash
curl -X POST http://127.0.0.1:8000/dialog -H "Content-Type: application/json" -d '{"user_text":"Hola, perdí mi tarjeta"}'
```

## Entrenar Banking77
```bash
python -m src.models.train_intent --model_name bert-base-uncased
```
