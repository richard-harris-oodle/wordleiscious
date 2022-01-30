from typing import Iterable
from importlib import resources
import json
import pandas as pd

DATA_PACKAGE = __package__ + ".data"


def answers() -> Iterable[str]:
    with resources.open_text(DATA_PACKAGE, "answers.json") as fp:
        yield from json.load(fp=fp)


def allowed_guesses() -> Iterable[str]:
    with resources.open_text(DATA_PACKAGE, "allowed_guesses.json") as fp:
        yield from json.load(fp=fp)


def all_words() -> Iterable[str]:
    yield from answers()
    yield from allowed_guesses()


def weights() -> pd.DataFrame:
    with resources.open_text(DATA_PACKAGE, "weights.json.gzip") as path_or_buf:
        return pd.read_json(path_or_buf=path_or_buf)
