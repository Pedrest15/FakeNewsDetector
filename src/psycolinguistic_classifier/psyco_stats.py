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


def get_psychometric_stat(doc, feature):

    values = [
        getattr(token._, feature)
        for token in doc
        if getattr(token._, feature) != 0
    ]

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
        'sum' : np.round(np.sum(values), 3),
        'vardiff' : np.round( np.var(np.diff(values), ddof=1), 4),
        'cv': np.round(np.std(values, ddof=1) / np.mean(values), 4) if np.mean(values) != 0 else 0
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
    Doc.set_extension(feature, getter=lambda doc, f=feature: get_psychometric_stat(doc, f))

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


class BasicRule:

    def apply(self, doc):
        raise NotImplementedError()

class Rule(BasicRule):

    def __init__(self, next_rule=None):
        self.next = next_rule

    def check(self, doc):
        stop = self.apply(doc)
        if not stop and self.next:
            self.next.check(doc)
        return doc


class BranchingRule(BasicRule):

    def __init__(self, if_true=None, if_false=False):
        self.branch_true = if_true
        self.branch_false = if_false

    def check(self, doc):
        if self.apply(doc):
            return self.branch_true.check(doc) if self.branch_true else 'classified_true'
        else:
            return self.branch_false.check(doc) if self.branch_false else 'classified_fake'


class HighConcreteness(Rule):

    def __init__(self, threshold=6.0, next_rule=None):
        super().__init__(next_rule)
        self.threshold = threshold

    def apply(self, doc):
        if doc._.concreteness['max'] > self.threshold:
            doc._.asserts.append('true')
        else:
            doc._.asserts.append('fake')

        return False

class HighAgeOfAcquisition(Rule):

    def __init__(self, threshold=8.0, next_rule=None):
        super().__init__(next_rule)
        self.threshold = threshold

    def apply(self, doc):
        if doc._.concreteness['max'] > self.threshold:
            doc._.asserts.append('true')
        else:
            doc._.asserts.append('fake')

        return False

class HighImagery(Rule):

    def __init__(self, threshold=6.0, next_rule=None):
        super().__init__(next_rule)
        self.threshold = threshold

    def apply(self, doc):
        if doc._.concreteness['max'] > self.threshold:
            doc._.asserts.append('true')
        else:
            doc._.asserts.append('fake')

        return False

class HighSubjectivity(Rule):

    def __init__(self, threshold=6.0, next_rule=None):
        super().__init__(next_rule)
        self.threshold = threshold

    def apply(self, doc):
        if doc._.concreteness['max'] > self.threshold:
            doc._.asserts.append('true')
        else:
            doc._.asserts.append('fake')

        return False


chain = HighConcreteness(
    next_rule=HighSubjectivity(
        next_rule=HighAgeOfAcquisition(
            next_rule=HighImagery()
        )
    )
)


def psycometric_classification(text : str, nlp=nlp, rule=chain):

    doc = nlp(text)
    chain.check(doc)
    counter = Counter(doc._.asserts)

    return 'fake' if (counter['fake'] > counter['true']) else 'true'








