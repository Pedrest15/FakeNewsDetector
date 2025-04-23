from FakeNewsDetector.src.ratio_classifier.ratio_classifier import RatioClassifier
import os
import csv

def classify_folder(folders: dict[str, str]):
    results = []
    classifier = RatioClassifier()

    for true_class, folder in folders.items():
        for file_name in os.listdir(folder):
            if file_name.endswith(".txt"):
                full_path = os.path.join(folder, file_name)
                with open(full_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    sys_class = classifier.news_classifier(text=text)
                    results.append([file_name, true_class, sys_class])

    return results

def make_results_file(file_name: str, results: list):
    with open(file_name+".csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Arquivo", "Classe Verdadeira", "Classe Prevista"])
        writer.writerows(results)

if __name__ == '__main__':
    full_text_folders = {
        "FAKE": r"Fake.br-Corpus-master\full_texts\fake",
        "REAL": r"Fake.br-Corpus-master\full_texts\true"
    }

    size_normalized_folders = {
        "FAKE": r"Fake.br-Corpus-master\size_normalized_texts\fake",
        "REAL": r"Fake.br-Corpus-master\size_normalized_texts\true"
    }

    results_full_text = classify_folder(folders=full_text_folders)
    results_size_normalized_text = classify_folder(folders=size_normalized_folders)    

    make_results_file(file_name="full_text_results", results=results_full_text)
    make_results_file(file_name="size_normalized_text_results", results=results_size_normalized_text)

    print("Classificação Concluída.")

