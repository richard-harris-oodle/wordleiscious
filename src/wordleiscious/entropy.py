from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from wordleiscious.words import all_words, answers


class Solver:
    def __init__(self, candidate: pd.Series, allowed_guess: pd.Series):
        self.candidate = candidate
        self.allowed_guess = allowed_guess

    @classmethod
    def from_candidates_and_guesses(
        cls, candidates: Iterable[str], allowed_guesses: Iterable[str]
    ):

        return cls(
            candidate=pd.Series(name="candidate", data=candidates),
            allowed_guess=pd.Series(name="guess", data=allowed_guesses),
        )

    def with_guess(self, guess: str, outcome: str) -> "Solver":
        remaining = Solver.remaining(
            candidate=self.candidate,
            guess=pd.Series(index=self.candidate.index, data=guess, name="guess"),
            outcome=pd.Series(index=self.candidate.index, data=outcome, name="outcome"),
        )
        candidate = self.candidate[remaining].copy()
        allowed_guess = self.allowed_guess[self.allowed_guess != guess].copy()
        return Solver(candidate=candidate, allowed_guess=allowed_guess)

    @staticmethod
    def remaining(
        candidate: pd.Series, guess: pd.Series, outcome: pd.Series
    ) -> pd.Series:
        n = candidate.str.len().max()

        remaining = pd.Series(index=candidate.index, name="remaining", data=True)

        remaining &= remaining[candidate != guess]

        cs = [candidate.str[i] for i in range(n)]

        for i in range(n):
            c = cs[i]
            o = outcome.str[i]
            g = guess.str[i]

            green_mask = o == "🟩"
            yellow_mask = o == "🟨"
            black_mask = o == "🟦"

            remaining[green_mask] &= (c == g)[green_mask]

            has_g = pd.Series(index=candidate.index, data=False)
            for ii in range(n):
                has_g |= cs[ii] == g

            remaining[yellow_mask] &= has_g[yellow_mask]

            remaining[black_mask] &= ~has_g[black_mask]

        return remaining

    @staticmethod
    def outcome_after_guess(candidate: pd.Series, guess: pd.Series) -> pd.DataFrame:

        n = candidate.str.len().max()

        outcome_df = pd.merge(candidate, guess, how="cross")

        outcome_df["outcome"] = ""

        cs = [outcome_df.candidate.str[i] for i in range(n)]

        for i in range(n):

            g = outcome_df.guess.str[i]
            c = cs[i]

            green_mask = g == c

            yellow_mask = pd.Series(index=outcome_df.index, data=False)
            for ii in range(n):
                yellow_mask |= g == cs[ii]
            yellow_mask &= ~green_mask

            black_mask = ~green_mask & ~yellow_mask

            outcome_df.loc[green_mask, "outcome"] += "🟩"
            outcome_df.loc[yellow_mask, "outcome"] += "🟨"
            outcome_df.loc[black_mask, "outcome"] += "🟦"

        return outcome_df

    @staticmethod
    def scalar_outcome_after_guess(candidate: str, guess: str) -> str:
        single_candidate = pd.Series(data=[candidate], name="candidate")
        single_guess = pd.Series(data=[guess], name="guess")
        outcome_df = Solver.outcome_after_guess(
            candidate=single_candidate, guess=single_guess
        )
        return outcome_df.outcome.values[0]

    @staticmethod
    def _ensemble_entropy(outcome: pd.Series) -> float:

        outcome_prob = outcome.value_counts()
        outcome_prob /= outcome_prob.sum()

        return (-outcome_prob * np.log2(outcome_prob)).sum()

    @staticmethod
    def evaluate(candidate: pd.Series, guess: pd.Series):
        outcome_df = Solver.outcome_after_guess(
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
            for guess_chunk in tqdm(guess_chunks, leave=False)
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

    words = list(all_words())

    def _pre_display(guess: str, outcome: str):
        print(f"guess:'{guess}', outcome:{outcome}", end="")

    def _post_display(s: Solver):
        print(f", remaining candidates:{s.candidate.size}")

    first_guess = "tares"

    original_solver = Solver.from_candidates_and_guesses(
        candidates=words, allowed_guesses=words
    )

    for solution in answers():

        first_outcome = Solver.scalar_outcome_after_guess(
            candidate=solution, guess=first_guess
        )

        _pre_display(guess=first_guess, outcome=first_outcome)
        s = original_solver.with_guess(guess=first_guess, outcome=first_outcome)
        _post_display(s=s)

        while True:
            best_guess = s.guess()

            best_outcome = Solver.scalar_outcome_after_guess(
                candidate=solution, guess=best_guess
            )
            _pre_display(guess=best_guess, outcome=best_outcome)
            if best_outcome == "🟩🟩🟩🟩🟩":
                break
            s = s.with_guess(guess=best_guess, outcome=best_outcome)
            _post_display(s=s)

        print("\n" * 2)


if __name__ == "__main__":
    main()
