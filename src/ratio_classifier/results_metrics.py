import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def calculate_metrics(file_name: str):
    df = pd.read_csv(file_name)

    # Padroniza as classes
    df["Classe Verdadeira"] = df["Classe Verdadeira"].str.lower()
    df["Classe Prevista"] = df["Classe Prevista"].str.lower()

    # Metricas
    print(f"========= Relatório de classificação p/ {file_name} ==================\n")
    print(classification_report(df["Classe Verdadeira"], df["Classe Prevista"], target_names=["fake", "true"]))

    # Matriz de confusao
    print("Matriz de confusão:")
    print(confusion_matrix(df["Classe Verdadeira"], df["Classe Prevista"], labels=["fake", "true"]))
    print("\n========================================================================")


if __name__ == '__main__':
    files = [
        r"ratio_classifier\full_text_results.csv",
        r"ratio_classifier\size_normalized_text_results.csv"
    ]

    for file in files:
        calculate_metrics(file_name=file.split("\\")[-1])