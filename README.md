# Clasificador de Documentos Legales

Sistema de clasificación automática de documentos judiciales utilizando modelos de lenguaje pre-entrenados.

## Descripción

Este proyecto implementa un clasificador de textos legales basado en transformers que identifica el tipo de documento judicial entre las categorías: Embargo, Desembargo, Requerimiento y Oficio.

## Requisitos

- Python 3.10+ (en particular se usó 3.13)
- Transformers
- PyTorch
- scikit-learn
- pandas

## Instalación

```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

```
embargos-ia/
├── data/
│   ├── oficios_judiciales_200.csv
│   └── synthetic_legal_dataset.csv
├── models/
│   └── distilbeto_oficios/
├── src/
│   ├── train.py
│   ├── infer.py
│   └── model_utils.py
├── results/
├── requirements.txt
└── README.md
```

## Uso

### Entrenamiento

Para entrenar el modelo con el dataset:

```bash
python src/train.py
```

El modelo entrenado se guardará en `models/distilbeto_oficios_v1/` junto con:
- Pesos del modelo (`model.safetensors`)
- Tokenizer
- Métricas de evaluación (`metrics.json`)
- Mapeo de etiquetas (`label2id.json`)

### Inferencia

Para realizar predicciones sobre nuevos textos:

```bash
python src/infer.py
```

Los resultados se guardarán en formato JSON con la siguiente estructura:

```json
{
  "texto": "El juzgado ordena el levantamiento de las medidas cautelares...",
  "prediccion": {
    "tipo_documento": "Desembargo",
    "probabilidad": 0.94
  }
}
```

## Modelo

**Modelo base:** `dccuchile/distilbert-base-spanish-uncased`

**Arquitectura:**
- DistilBERT pre-entrenado en español
- Capa de clasificación con 4 clases
- 66M parámetros

**Hiperparámetros de entrenamiento:**
- Optimizer: AdamW
- Learning rate: 5e-5
- Batch size: 16 (entrenamiento), 32 (evaluación)
- Épocas: 5
- Weight decay: 0.01
- Scheduler: Linear con warmup

## Métricas

El modelo se evalúa utilizando las siguientes métricas:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)

## Dataset

El dataset de entrenamiento contiene 1200 "documentos" legales clasificados en 4 categorías, el dataset fue generado sintéticamente para incluir 300 documentos por categoría para entrenamiento, construido a partir de sinónimos y frases comunes en la jerga legal. Para elaborar las oraciones se consultaron fuentes como el Diccionario de la Lengua Española de la RAE, que indica que embargo tiene sinónimos como incautación, confiscación, retención, secuestro, requisa y decomiso y que requerimiento puede equipararse a petición, solicitud, exigencia, demanda, orden, mandato, aviso y requisitoria. Para la clase desembargo se tomaron expresiones equivalentes como liberación, levantamiento del embargo, desbloqueo, liberar una cuenta, liberar bienes embargados y desaplicación del embargo, mientras que la definición oficial de traslado tiene variantes como notificación, remisión de documentos o solicitud de traslado.

El archivo CSV resultante contiene columnas texto y tipo_documento, con nombres de personas, cuentas, fechas y ubicaciones variadas para simular oficios reales en las categorías de:

- Embargo
- Desembargo
- Requerimiento
- Traslado

La división de datos para el modelo será inicialmente:
- Entrenamiento: 80%
- Validación: 20%

esta proporcion puede modificarse en la función load_data

## Estrategia de aprendizaje continuo:
Reentrenamiento del modelo agregando los documentos que fueron clasificados de manera erronea pero con la etiqueta correcta. Los psos para ellos son sencillos:
- Agregue al synthetic_legal_dataset.csv los documentos separado por coma con su correspondiente etiqueta (ej. Traslado, Embargo, etc.)
Si desea modificar la variable:

```python
OUTPUT_DIR = "models/distilbeto_oficios_v..."
```
De esta forma podrá tener disponibles diferentes versiones del modelo entrenado o incluso entrenar diferentes modelos simplemente modificando la variable:

```python
MODEL_NAME = "dccuchile/distilbert-base-spanish-uncased"
```

de manera opcional puede modificar también:

```python
DATA_PATH = os.path.join("data", "synthetic_legal_dataset.csv")
```

y entrenar con cualquier otro dataset que venga estructurado de la forma:

texto, tipo_documento

Al haber realizado los cambios que considere pertinentes simplemente reentrene el modelo:

```bash
python src/train.py
```


## Mejoras Futuras

- Aumentar el dataset con más ejemplos reales
- Probar con modelos más grandes (BETO, RoBERTa)
- Implementar data augmentation específico para textos legales
- Agregar explicabilidad de predicciones con LIME o SHAP

## Autor

Desarrollado como prueba técnica para el rol de Ingeniero de Desarrollo en IA por Rafael Castro.
