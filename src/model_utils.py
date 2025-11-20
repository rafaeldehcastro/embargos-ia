import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "models/distilbeto_oficios"

def load_model(model_dir: str = MODEL_DIR):
    """
    Carga el tokenizer, modelo y mapeo de etiquetas.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    with open(os.path.join(model_dir, "label2id.json"), "r", encoding="utf-8") as f:
        label2id = json.load(f)

    id2label = {int(v): k for k, v in label2id.items()}

    return tokenizer, model, id2label


def predict(texts, tokenizer, model, id2label, max_length=128):
    """
    Realiza inferencia sobre una lista de textos y devuelve las predicciones en formato JSON.
    """
    model.eval()
    results = []

    with torch.no_grad():
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        outputs = model(**enc)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        confs, preds = torch.max(probs, dim=-1)

        for text, pred_id, conf in zip(texts, preds, confs):
            label = id2label[pred_id.item()]
            prob = float(conf.item())

            results.append(
                {
                    "texto": text,
                    "prediccion": {
                        "tipo_documento": label,
                        "probabilidad": round(prob, 4),
                    },
                }
            )

    return results