import sys
from text_features import nlp, extrair_metricas_otimizado

def classify_by_rules(text: str) -> int:
    doc = nlp(text)
    m = extrair_metricas_otimizado(doc)
    if m["informalidade"] <= 1.18:
        return 1 if m["brunet"] <= 10.21 else 0
    else:
        return 1 if m["pausality"] <= 2.81 else 0

def main(text: str) -> str:
    label = classify_by_rules(text)
    return "TRUE" if label == 1 else "FALSE"

if __name__ == "__main__":
    texto = " ".join(sys.argv[1:])
    resultado = main(texto)
    print(resultado)
