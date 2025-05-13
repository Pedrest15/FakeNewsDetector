# @title command.py
import numpy as np
import operator
from spacy.tokens import Doc

from typing import (Any, Callable, Union)

class Command:
    def __init__(self, *args, **kwargs):
        pass

    def eval(self, doc : Doc) -> bool:
        raise NotImplementedError("Subclasses should implement this method.")

    def __call__(self, *args, **kwds):
        return self.eval(*args, **kwds)

class Comparison(Command):

    OPERATORS = {
        '==': operator.eq,
        '!=': operator.ne,
        '<': operator.lt,
        '>': operator.gt,
        '>=': operator.ge,
        '<=': operator.le,
    }

    STATS_FN = {
        'mean' : np.mean,
        'median' : np.median,
        'std' : np.std,
        'var' : np.var,
        'min' : np.min,
        'max' : np.max,
        'sum' : np.sum,
        'count' : len,
    }

    def __init__(self, stat : str,  field : str, operator : str, value):
        if operator not in self.OPERATORS:
            raise ValueError(f"Invalid operator: {operator}")
        if isinstance(stat, str) and stat not in self.STATS_FN:
            raise ValueError(f"Invalid stat operatior: {stat}")
        self.stat_fn = self.get_stat_function(stat)
        self.field = field
        self.operator = self.OPERATORS[operator]
        self.threshold = value

    def eval(self, doc : Doc) -> bool:
        if not hasattr(doc._, self.field):
            raise ValueError(f"Field '{self.field}' not found in the document.")
        values = getattr(doc._, self.field)
        value = self.stat_fn(values)
        return self.operator(value, self.threshold)

    def get_stat_function(self, stat):
        if isinstance(stat, str) and stat in self.STATS_FN:
            return self.STATS_FN[stat]
        elif callable(stat):
            return stat
        else:
            raise ValueError(f"Invalid stat function: {stat}. Must be one of {list(self.STATS_FN.keys())} or a callable.")

class AndCommand(Command):
    def __init__(self, *commands : Command):
        self.commands = commands

    def eval(self, doc : Doc) -> bool:
        return all(command.eval(doc) for command in self.commands)

class OrCommand(Command):

    def __init__(self, *commands : Command):
        self.commands = commands

    def eval(self, doc : Doc) -> bool:
        return any(command.eval(doc) for command in self.commands)


class BetweenCommand(Command):

    def __init__(self, stat: str, field: str, min_value, max_value):
        self.field = field
        self.min_value = min_value
        self.max_value = max_value
        self.stat_fn = self.get_stat_function(stat)

    def eval(self, doc : Doc) -> bool:
        if not hasattr(doc._, self.field):
            raise ValueError(f"Field '{self.field}' not found in the document.")
        values = getattr(doc._, self.field)
        value = self.stat_fn(values)
        return self.min_value <= value <= self.max_value

    def get_stat_function(self, stat):
        if isinstance(stat, str) and stat in Comparison.STATS_FN:
            return Comparison.STATS_FN[stat]
        elif callable(stat):
            return stat
        else:
            raise ValueError(f"Invalid stat function: {stat}. Must be one of {list(Comparison.STATS_FN.keys())} or a callable.")

class BranchingCommand(Comparison):

    def __init__(self, condition: Comparison, if_true: Union[bool, Command, Any]=True, if_false: Union[bool, Command, Any]=False):
        self.condition = condition
        self.if_true = if_true
        self.if_false = if_false

    def eval(self, doc : Doc) -> bool:
        if self.condition.eval(doc):
            return self.if_true.eval(doc) if isinstance(self.if_true, Command) else self.if_true
        else:
            return self.if_false.eval(doc) if isinstance(self.if_false, Command) else self.if_false
