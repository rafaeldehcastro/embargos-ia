import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "dccuchile/distilbert-base-spanish-uncased"
#DATA_PATH = os.path.join("data", "oficios_judiciales_200.csv")
DATA_PATH = os.path.join("data", "synthetic_legal_dataset.csv")
OUTPUT_DIR = "models/distilbeto_oficios_v1"

def load_data():
    """
    Carga y prepara el dataset de documentos legales para entrenamiento.
    
    Returns:
        tuple: (train_df, val_df, label2id, id2label)
            - train_df: DataFrame de entrenamiento (80%)
            - val_df: DataFrame de validación (20%)
            - label2id: Diccionario que mapea etiquetas a IDs numéricos
            - id2label: Diccionario que mapea IDs numéricos a etiquetas
    """
    df = pd.read_csv(DATA_PATH)
    # columnas: texto, tipo_documento
    # Crear mapping etiqueta -> id
    labels = sorted(df["tipo_documento"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    df["label"] = df["tipo_documento"].map(label2id)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )
    return train_df, val_df, label2id, id2label

class OficiosDataset:
    """
    Dataset personalizado para textos de oficios judiciales.
    Convierte textos en tensores tokenizados para el modelo.
    """
    def __init__(self, df, tokenizer, max_length=128):
        """
        Inicializa el dataset con textos y etiquetas.
        
        Args:
            df: DataFrame con columnas 'texto' y 'label'
            tokenizer: Tokenizer de Hugging Face para procesar textos
            max_length: Longitud máxima de secuencia (default: 128)
        """
        self.texts = df["texto"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Retorna el número total de muestras en el dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retorna una muestra tokenizada del dataset.
        
        Args:
            idx: Índice de la muestra
            
        Returns:
            dict: Diccionario con 'input_ids', 'attention_mask' y 'labels'
        """
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in enc.items()}
        item["labels"] = int(self.labels[idx])
        return item

def compute_metrics(eval_pred):
    """
    Calcula métricas de evaluación para el modelo.
    
    Args:
        eval_pred: Tupla (logits, labels) de las predicciones
        
    Returns:
        dict: Diccionario con accuracy, precision, recall y f1-score
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def main():
    """
    Función principal que ejecuta el flujo completo de entrenamiento:
    1. Carga datos y crea datasets
    2. Inicializa modelo y tokenizer
    3. Configura parámetros de entrenamiento
    4. Entrena el modelo
    5. Evalúa y guarda resultados
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Carga de datos
    train_df, val_df, label2id, id2label = load_data()

    # 2. Tokenizer y modelo
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    train_dataset = OficiosDataset(train_df, tokenizer)
    val_dataset = OficiosDataset(val_df, tokenizer)

    # 3. Configuración de entrenamiento
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",   # eval cada época
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        learning_rate=5e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 4. Entrenar (Trainer usa AdamW + scheduler linear por defecto)
    trainer.train()

    # 5. Evaluar
    metrics = trainer.evaluate()
    print("Métricas finales:", metrics)

    # Guardar métricas en JSON
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 6. Guardar modelo y tokenizer
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Guardar mapping de etiquetas
    with open(os.path.join(OUTPUT_DIR, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2, ensure_ascii=False)

    print("Entrenamiento finalizado y modelo guardado en:", OUTPUT_DIR)


if __name__ == "__main__":
    main()