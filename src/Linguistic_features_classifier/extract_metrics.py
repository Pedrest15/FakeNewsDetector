import argparse
from text_features import nlp, extrair_metricas_otimizado

def classify_by_rules(text: str) -> int:
    """
    Árvore de decisão:
      ├─ informalidade ≤ 1.18
      │   ├─ brunet ≤ 10.21  → FAKE (1)
      │   └─ brunet > 10.21  → REAL (0)
      └─ informalidade > 1.18
          ├─ pausality ≤ 2.81 → FAKE (1)
          └─ pausality > 2.81 → REAL (0)
    """
    doc = nlp(text)
    m = extrair_metricas_otimizado(doc)

    if m["informalidade"] <= 1.18:
        return 1 if m["brunet"] <= 10.21 else 0
    else:
        return 1 if m["pausality"] <= 2.81 else 0

def main():
    parser = argparse.ArgumentParser(
        description="Classifica um texto como REAL ou FAKE via regras spaCy+Pyphen"
    )
    parser.add_argument(
        "text",
        nargs="+",
        help="O texto a ser classificado (entre aspas se tiver espaços)"
    )
    args = parser.parse_args()
    texto = " ".join(args.text)
    label = classify_by_rules(texto)
    print("FAKE" if label == 1 else "REAL")

if __name__ == "__main__":
    main()
