from sklearn.metrics import (
    classification_report
)

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import random
import re
import os

from matplotlib import pyplot as plt
import seaborn as sns

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .psyco_stats import (
     nlp,
     bch6,
     psycometric_classification,
     get_psychometric_features,
     get_psychometric_stats,
)


def read_file(file):
        *_, label, name = file.parts
        return {'id' : int(file.stem), 'name': name, 'content': file.read_text(), 'label' : label}

def apply_nlp(doc):
    doc['content'] = nlp(doc['content']) if isinstance(doc['content'], str) else doc['content']
    return doc

if __name__ == "__main__":

    max_workers = min(32, os.cpu_count() + 4)
    print('cpu count', os.cpu_count())
    print(f'{max_workers=}')

    files = list()
    files.extend(Path('./Fake.br-Corpus/full_texts/true').glob('*.txt'))
    files.extend(Path('./Fake.br-Corpus/full_texts/fake').glob('*.txt'))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        data = list(executor.map(read_file, files))

    random.shuffle(data)

    print(f'Read {len(data)} files.')

    len(files)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        data = list(executor.map(apply_nlp, data))

    Docs_train, Docs_test = train_test_split(data, train_size=0.7)

    print(len(Docs_train), len(Docs_test))

    y_trues = [doc['label'] for doc in Docs_train]
    y_preds = np.array(
        [ 'fake' if bch6.eval(exemplo['content']) else 'true' for exemplo in Docs_train]
    )

    print(
        classification_report(y_trues, y_preds)
    )

    y_test_trues = [doc['label'] for doc in Docs_test]
    y_test_preds = np.array(
        [ 'fake' if bch6.eval(exemplo['content']) else 'true' for exemplo in Docs_test]
    )

    print(
        classification_report(y_test_trues, y_test_preds)
    )
