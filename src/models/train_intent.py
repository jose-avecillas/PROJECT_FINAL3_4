import argparse, json, numpy as np, evaluate
from pathlib import Path
from datasets import load_dataset
from transformers import (AutoTokenizer, DataCollatorWithPadding,
                          AutoModelForSequenceClassification, TrainingArguments, Trainer)
from sklearn.metrics import f1_score

OUT_DIR = Path(__file__).resolve().parents[2] / "models" / "trained_models" / "banking77_bert"

def main(model_name: str = "bert-base-uncased", output_dir: Path = OUT_DIR):
    ds = load_dataset("banking77")
    label_list = ds["train"].features["label"].names
    num_labels = len(label_list)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True)

    ds_tok = ds.map(tokenize_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": metric_acc.compute(references=labels, predictions=preds)["accuracy"],
            "f1_macro": metric_f1.compute(references=labels, predictions=preds, average="macro")["f1"],
            "f1_weighted": f1_score(labels, preds, average="weighted")
        }

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    args = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        metric_for_best_model="f1_macro",
        load_best_model_at_end=True,
        logging_steps=50,
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds_tok["train"],
                      eval_dataset=ds_tok["validation"], tokenizer=tokenizer,
                      data_collator=data_collator, compute_metrics=compute_metrics)
    trainer.train()
    metrics = trainer.evaluate()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir); tokenizer.save_pretrained(output_dir)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print({"saved_to": str(output_dir), "metrics": metrics})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="bert-base-uncased")
    args = parser.parse_args()
    main(model_name=args.model_name)
