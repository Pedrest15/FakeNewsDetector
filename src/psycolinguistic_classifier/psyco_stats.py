import pandas as pd
import numpy as np
import nltk
import spacy
import random
import spacy
from collections import Counter
from spacy.language import Language
from spacy.tokens import Doc, Token

from pathlib import Path

from .commands import (
    Comparison,
    BranchingCommand,
)

source_file = r"src\psycolinguistic_classifier\BP.csv"

#if not source_file.exists():
#    raise FileNotFoundError('Check the psycholinguistic file location!')

psylin = pd.read_csv(
    source_file,
    sep=',',
    header=0,
    names=[
        'word',
        'grammatical_category',
        'concreteness',
        'subjectivity',
        'imagery',
        'aoa',
        'log_frequency',
        'frequency'
    ]
)

psylin.set_index('word', inplace=True)

vocabulary = psylin.to_dict(orient='index')

def get_psychometric_features(doc : Doc, feature : str):
    values = [
        getattr(token._, feature)
        for token in doc
        if getattr(token._, feature) != 0
    ]
    return values

def get_psychometric_stats(doc : Doc, feature : str):

    values = get_psychometric_features(doc, feature)

    if (not values) | (all(v == 0 for v in values)):
        stats = {
            'mean' : 0.0,
            'median' : 0.0,
            'max' : 0.0,
            'min' : 0.0,
        }

    stats = {
        'mean' : np.round(np.mean(values), 4),
        'median' : np.round(np.median(values), 4),
        'std' : np.round(np.std(values, ddof=1), 3),
        'max' : np.max(values),
        'min' : np.min(values),
        # 'sum' : np.round(np.sum(values), 3),
        # 'vardiff' : np.round( np.var(np.diff(values), ddof=1), 4),
        # 'cv': np.round(np.std(values, ddof=1) / np.mean(values), 4) if np.mean(values) != 0 else 0
    }

    return stats

features = [
    'concreteness',
    'subjectivity',
    'imagery',
    'aoa'
]


for feature in features:
    if Token.has_extension(feature):
        Token.remove_extension(feature)
    Token.set_extension(feature, default=0)

    if Doc.has_extension(feature):
        Doc.remove_extension(feature)
    Doc.set_extension(feature, getter=lambda doc, f=feature: get_psychometric_features(doc, f))

for feature in features:
    assert Token.has_extension(feature)
    assert Doc.has_extension(feature)
    print(feature)

if not Doc.has_extension('label'):
    print('Definindo propriedade: label')
    Doc.set_extension( 'label', default=None )

if not Doc.has_extension('asserts'):
    Doc.set_extension('asserts', default=list())


@Language.component('annotate_psychometrics')
def annotate_psychometrics(doc):
    for token in doc:
        lower = token.text.lower()
        if lower in vocabulary:
            values = vocabulary[lower]
            token._.concreteness = values['concreteness']
            token._.subjectivity = values['subjectivity']
            token._.imagery = values['imagery']
            token._.aoa = values['aoa']

    return doc


nlp = spacy.blank('pt')

# nlp.add_pipe('sentencizer')
nlp.add_pipe('annotate_psychometrics')

# conditions = [
#         Comparison(
#             stat='max',
#             field='concreteness',
#             operator='>',
#             value=6.0
#         ),
#         Comparison(
#             stat='max',
#             field='aoa',
#             operator='>',
#             value=8.0
#         ),
#         Comparison(
#             stat='max',
#             field='imagery',
#             operator='>',
#             value=6.0
#         ),
#         Comparison(
#             stat='max',
#             field='subjectivity',
#             operator='>',
#             value=6.0
#         ),
# ]

cmd1 = Comparison(
    stat='max',
    field='aoa',
    operator='<=',
    value=8.18
)

cmd2 = Comparison(
    stat='min',
    field='imagery',
    operator='<=',
    value=3.21
)

cmd3 = Comparison(
    stat='max',
    field='imagery',
    operator='<=',
    value=5.68
)

cmd4 = Comparison(
    stat='max',
    field='imagery',
    operator='<=',
    value=5.98
)

cmd5 = Comparison(
    stat='max',
    field='aoa',
    operator='<=',
    value=8.07
)

cmd6 = Comparison(
    stat='min',
    field='imagery',
    operator='<=',
    value=3.00
)

bch1 = BranchingCommand(
    condition=cmd1,
    if_true=True,
    if_false=False
)

bch2 = BranchingCommand(
    condition=cmd2,
    if_true=bch1,
    if_false=True # Estranho!
)

bch3 = BranchingCommand(
    condition=cmd3,
    if_true=True,
    if_false=False
)

bch4 = BranchingCommand(
    condition=cmd4,
    if_true=True,
    if_false=False
)

bch5 = BranchingCommand(
    condition=cmd5,
    if_true=bch4,
    if_false=bch3
)

bch6 = BranchingCommand(
    condition=cmd6,
    if_true=bch5,
    if_false=bch2
)

def psycometric_classification(in_doc : str, nlp=nlp, ruler=bch6):

    doc =  nlp(in_doc) if isinstance(in_doc, str) else in_doc

    return 'fake' if ruler.eval(doc) else 'true'