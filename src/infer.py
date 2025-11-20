import os
import json

from model_utils import load_model, predict

MODEL_DIR = "models/distilbeto_oficios"
RESULTS_DIR = "results"

def main():
    tokenizer, model, id2label = load_model(MODEL_DIR)

    nuevos_textos = [
        "El juzgado ordena el levantamiento de las medidas cautelares sobre la cuenta del demandado.",
        "Se decreta embargo de los bienes muebles ubicados en la dirección indicada.",
        "El despacho requiere al banco remitir certificación actualizada de los saldos.",
        "Se traslada a la parte demandada copia del escrito presentado por la parte actora.",
        "Se solicita al juzgado informar sobre el estado actual del proceso ejecutivo.",
        "El juzgado dispone el levantamiento de la medida de embargo sobre la cuenta 364007.",
        "El juzgado autoriza el desembargo de los fondos retenidos en Bancolombia.",
        "El despacho judicial solicita los movimientos financieros de Martínez.",
        "El despacho judicial decreta embargo de los ahorros en Banco de Occidente.",
        "Se requiere al banco reporte actualizado de las cuentas afectadas.",
        "Se remite oficio al Juzgado 982625 de Ejecución para la continuación del trámite.",
        "Se requiere al banco reporte actualizado de las cuentas afectadas.",
        "Notifíquese a la entidad financiera la liberación de los recursos de Ramírez.",
        "Se requiere al banco BBVA remitir los extractos de la cuenta 329408.",
        "Se solicita al banco Davivienda enviar los estados de cuenta de los últimos tres meses.",
    ]

    resultados = predict(
        texts=nuevos_textos,
        tokenizer=tokenizer,
        model=model,
        id2label=id2label
    )

    # Guardar JSON
    out_path = os.path.join(RESULTS_DIR, "inferencias_5_ejemplos_2.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print("JSON de inferencias guardado en:", out_path)
    print(json.dumps(resultados, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()