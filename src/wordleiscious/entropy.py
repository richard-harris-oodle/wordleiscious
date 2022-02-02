from typing import Iterable

import numpy as np
import pandas as pd

from wordleiscious.outcome import (
    outcome_after_guess,
    scalar_outcome_after_guess,
    remaining,
    display,
)
from wordleiscious.words import all_words, answers


class Solver:
    def __init__(self, candidate: pd.Series, allowed_guess: pd.Series):
        self.candidate = candidate
        self.allowed_guess = allowed_guess

    @classmethod
    def from_candidates_and_guesses(
        cls,
        candidates: Iterable[str],
        allowed_guesses: Iterable[str],
    ):
        return cls(
            candidate=pd.Series(name="candidate", data=candidates),
            allowed_guess=pd.Series(name="guess", data=allowed_guesses),
        )

    def with_guess(
        self,
        guess: str,
        outcome: str,
        hard_mode: bool = False,
    ) -> "Solver":
        _remaining = remaining(
            candidate=self.candidate,
            guess=pd.Series(index=self.candidate.index, data=guess, name="guess"),
            outcome=pd.Series(index=self.candidate.index, data=outcome, name="outcome"),
        )
        candidate = self.candidate[_remaining].copy()
        if hard_mode:
            allowed_guess = self.allowed_guess[
                (self.allowed_guess != guess)
                & np.isin(self.allowed_guess, candidate.values)
            ]
        else:
            allowed_guess = self.allowed_guess[(self.allowed_guess != guess)]

        return Solver(candidate=candidate, allowed_guess=allowed_guess)

    @staticmethod
    def _ensemble_entropy(outcome: pd.Series) -> float:

        outcome_prob = outcome.value_counts()
        outcome_prob /= outcome_prob.sum()

        return (-outcome_prob * np.log2(outcome_prob)).sum()

    @staticmethod
    def evaluate(candidate: pd.Series, guess: pd.Series):
        outcome_df = outcome_after_guess(
            candidate=candidate,
            guess=guess,
        )

        return (
            outcome_df.groupby("guess")
            .outcome.apply(Solver._ensemble_entropy)
            .rename("entropy")
        )

    def evaluate_guesses(self) -> pd.Series:

        chunk_size = 100
        guess_chunks = [
            self.allowed_guess[i : i + chunk_size]
            for i in range(0, self.allowed_guess.size, chunk_size)
        ]

        return pd.concat(
            Solver.evaluate(candidate=self.candidate, guess=guess_chunk)
            for guess_chunk in guess_chunks
        )

    def best_guesses(self) -> pd.DataFrame:
        guess_scores_df = self.evaluate_guesses().reset_index()
        best_entropy = guess_scores_df.entropy.max()
        return guess_scores_df[guess_scores_df.entropy == best_entropy]

    def guess(self) -> str:
        if self.candidate.size == 1:
            return self.candidate.values[0]
        return self.best_guesses().sample().guess.values[0]


def main():

    candidates = list(all_words())
    words = list(all_words())

    def _pre_display(guess: str, outcome: str):
        print(display(guess=guess, outcome=outcome), end="")

    def _post_display(s: Solver):
        print(f", remaining candidates:{s.candidate.size}")

    first_guess = "tares"

    original_solver = Solver.from_candidates_and_guesses(
        candidates=candidates, allowed_guesses=words
    )

    for solution in answers():

        first_outcome = scalar_outcome_after_guess(
            candidate=solution, guess=first_guess
        )

        _pre_display(guess=first_guess, outcome=first_outcome)
        s = original_solver.with_guess(guess=first_guess, outcome=first_outcome)
        _post_display(s=s)

        while True:
            best_guess = s.guess()

            best_outcome = scalar_outcome_after_guess(
                candidate=solution, guess=best_guess
            )
            _pre_display(guess=best_guess, outcome=best_outcome)
            if best_outcome == "🟩🟩🟩🟩🟩":
                break
            s = s.with_guess(guess=best_guess, outcome=best_outcome, hard_mode=True)
            _post_display(s=s)

        print("\n" * 2)


if __name__ == "__main__":
    main()
