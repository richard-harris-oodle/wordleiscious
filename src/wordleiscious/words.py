from typing import Iterable, Dict
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


def word_frequency() -> pd.DataFrame:
    return pd.read_feather(path=resources.path(DATA_PACKAGE, "word_frequency.feather"))


def process_frequency_file():
    with resources.open_text(DATA_PACKAGE, "unigram_freq.csv") as filepath_or_buffer:
        unigram_freq_df = pd.read_csv(filepath_or_buffer=filepath_or_buffer)
    unigram_freq_df = unigram_freq_df[unigram_freq_df.word.str.len() == 5].reset_index(
        drop=True
    )
    unigram_freq_df.to_feather(
        path=resources.path(DATA_PACKAGE, "word_frequency.feather")
    )


def candidate_weights() -> Dict[str, float]:
    frequency_df = word_frequency().set_index("word")
    word_index = pd.Index(data=list(all_words()))

    weighted_candidate_df = (
        frequency_df.reindex(word_index)
        .fillna(frequency_df["count"].min())
        .rename(columns={"count": "weight"})
    )
    weighted_candidate_df["weight"] /= weighted_candidate_df["weight"].sum()

    return weighted_candidate_df.weight.to_dict()
