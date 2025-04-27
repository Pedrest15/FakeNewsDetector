import os
import pandas as pd
from git import Repo

def clone_corpus(dest_path: str = "data/Fake.br-Corpus") -> None:
    if not os.path.exists(dest_path):
        Repo.clone_from("https://github.com/roneysco/Fake.br-Corpus", dest_path)

def load_fake_true(data_dir: str = "data/Fake.br-Corpus/full_texts") -> pd.DataFrame:
    df = pd.DataFrame(columns=["noticia", "label"])
    for label, sub in [("fake", 1), ("true", 0)]:
        folder = os.path.join(data_dir, sub if isinstance(sub, str) else label)
        for fname in os.listdir(folder):
            text = open(os.path.join(folder, fname), "r", encoding="utf-8").read()
            df = df.append({"noticia": text, "label": sub}, ignore_index=True)
    return df
