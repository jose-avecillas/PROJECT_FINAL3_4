
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

1) Chatbot completamente funcional 
Qué es: un asistente bancario 24/7 que entiende texto, extrae datos relevantes, conversa con memoria, consulta un mini-conocimiento y ejecuta operaciones mock (saldo/transferencias) con trazabilidad y privacidad.

Cómo está construido en tu notebook:
•	NLU (intents):
o	Selector en caliente entre fallback por palabras clave, Zero-Shot multilingüe (XNLI) y Banking77 (EN).
o	Se controla vía GET/POST /intent_backend y se usa automáticamente dentro de /predict_intent.
•	NER (entidades):
o	spaCy (modelo es_core_news_sm) + regex para patrones bancarios (monto, cuenta, fecha).
o	Expuesto por /extract_entities.
•	Gestor de diálogo con memoria:
o	dialog_reply(text, ocr_text="") mantiene breve memoria (deque), consulta el mini-KG (límites/ETAs) y activa escalamiento si la confianza < umbral o el intent es crítico (p. ej., reclamos/fraude).
o	Expuesto por /chat.
•	Visión:
o	OCR con pytesseract y clasificación heurística de documentos (cheque/ID/voucher/otro).
o	Rutas /ocr y /classify_document.
•	Transacciones mock seguras:
o	/balance y /transfer validan límite diario, fondos, formato de monto y enmascaran PII en auditoría.
o	Puedes activar API-Key para proteger /transfer.
•	Privacidad y auditoría:
o	Enmascarado de emails y números largos en logs, y AUDIT_LOG con eventos clave.
Qué valida el “funcional”: puedes iniciar el servidor, obtener 200 en /health, clasificar intenciones/entidades, conversar y ejecutar una transferencia mock, con todo auditado y metrificado.

2) Pipeline de procesamiento multimodal (con /multimodal)
Qué es: un flujo que fusiona texto + imagen en una misma petición para mejorar la comprensión y completar slots.

Cómo funciona el endpoint /multimodal:
1.	Recibe text (Formulario) y file (imagen opcional).
2.	Si hay imagen, ejecuta OCR → ocr_text.
3.	Fusión: concatena text + ocr_text → fused_text.
4.	Llama a dialog_reply(fused_text, ocr_text=ocr_text) para generar la respuesta con:
o	NLU/NER sobre el texto fusionado,
o	KG para enriquecer (p. ej., límites),
o	escalamiento si aplica.
5.	Devuelve la respuesta, intent, confidence, entities, kg_info, fragmento de ocr_text y longitud de fused_text.
Utilidad práctica: el usuario puede subir un comprobante o identificación y complementar con texto; el bot extrae datos, entiende la intención y sugiere/ejecuta la acción (mock), reduciendo fricción y errores.

3) Interfaz web y API  (Gradio HTTP + FastAPI)
API FastAPI:
•	Endoints principales:
/health, /predict_intent, /extract_entities, /chat, /ocr, /classify_document, /multimodal, /balance, /transfer, /feedback, /feedback_stats, /audit_log, /intent_backend (GET/POST), /persist/* (si activaste persistencia).
•	Arranque estable: uvicorn en hilo, puerto libre automático y detección de BASE_URL.
•	Seguridad opcional: API-Key con header X-API-Key para proteger endpoints sensibles.

UI Gradio (HTTP):
•	Llama a la API por HTTP usando BASE_URL.
•	Tabs:
o	💬 Chat: conversa y muestra metadatos (intent, confianza, entidades, escalamiento).
o	💸 Transacciones: saldo y transferencia mock.
o	🧾 Documentos: OCR y clasificación cargando imagen.
o	📊 Métricas: envío de feedback y estadísticas.
o	🧪 Auditoría: últimos eventos (recortados).

•	Panel NLU: selecciona fallback/zero_shot/banking77 en caliente vía /intent_backend.

4) Sistema de evaluación automática 
Qué evalúa: precisión del NLU (intents) y latencia promedio contra una muestra del dataset Banking77 (inglés).

Cómo lo hace:
1.	Carga banking77 (Hugging Face) y toma N ejemplos aleatorios del split test.
2.	Mapea la etiqueta original a tus intents canónicos (función de mapeo heurístico).
3.	Para cada texto, llama a /predict_intent (usa el backend activo real: zero-shot/banking77/fallback).
4.	Calcula:
o	Accuracy global (ok/total),
o	Accuracy por clase,
o	Latencia promedio (ms).
5.	Guarda un resumen JSON en /content/eval_intents_summary.json.

Notas importantes:
•	Si tu backend activo es zero-shot y pruebas con Inglés (Banking77), puede variar la precisión pero es una sanity check útil.
•	Para español, te conviene armar un mini-dataset propio y reutilizar el mismo script (simplemente cambiando los ejemplos).
•	Ajustes: sube N si usas GPU; fija semilla si necesitas reproducibilidad.

Beneficio: la UI te da una demo completa listada contra la misma API que expondrías a otros canales (web/app móvil/contact center).

5) Métricas de satisfacción del cliente  (feedback + /feedback_stats)
Qué hay:
•	Endpoint /feedback: recibe score (1–5) y helpful (bool) más un meta.flow (ej. “transfer”).
•	Endpoint /feedback_stats: devuelve conteo, promedio de score y tasa de “helpful”.
•	UI Gradio incluye una pestaña 📊 Métricas para enviar feedback y ver estadísticas.

Cómo se calcula:
•	En memoria (lista FEEDBACK), con funciones:
o	record_feedback(score, helpful, meta) → agrega registro.
o	feedback_stats() → {count, avg_score, helpful_rate}.
Para qué sirve:
•	Medir CSAT y utilidad percibida por flujo (transfer, reclamo, etc.).
•	Detectar caídas de satisfacción tras cambios de modelo o reglas.
•	Integrable con dashboards externos (en roadmap).

Resumen de valor para el proyecto
•	End-to-end: desde comprensión (NLU/NER) y fusión multimodal hasta acciones (mock) y UI.
•	Conmutable y auditable: puedes cambiar el motor de intents en vivo y seguir la trazabilidad.
•	Medible y mejorable: cuentas con evaluación automática y métricas de satisfacción.
•	Extensible: fácil de reforzar con modelos finetuned, RAG/LLM, OCR mejorado, seguridad y MLOps.

