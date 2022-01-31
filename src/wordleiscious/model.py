import numpy as np
import pandas as pd
from typing import Dict, List, Iterable

from wordleiscious.words import candidate_weights, answers


class Solver:
    def __init__(self, candidate_weight: pd.Series, allowed_guess: pd.Series):
        self.candidate_weight = candidate_weight
        self.allowed_guess = allowed_guess

    @classmethod
    def from_weights_and_guesses(
        cls, candidate_weights: Dict[str, float], allowed_guesses: Iterable[str]
    ):

        return cls(
            candidate_weight=pd.Series(name="weight", data=candidate_weights),
            allowed_guess=pd.Series(name="guess", data=allowed_guesses),
        )

    def with_guess(self, guess: str, outcome: str) -> "Solver":
        remaining = Solver.remaining(
            candidate=self.candidate_weight.index.to_series(),
            guess=pd.Series(
                index=self.candidate_weight.index, data=guess, name="guess"
            ),
            outcome=pd.Series(
                index=self.candidate_weight.index, data=outcome, name="outcome"
            ),
        )
        candidate_weight = self.candidate_weight[remaining].copy()
        allowed_guess = self.allowed_guess[self.allowed_guess != guess].copy()
        return Solver(candidate_weight=candidate_weight, allowed_guess=allowed_guess)

    @staticmethod
    def remaining(
        candidate: pd.Series, guess: pd.Series, outcome: pd.Series
    ) -> pd.Series:
        n = candidate.str.len().max()

        remaining = pd.Series(index=candidate.index, name="remaining", data=True)

        remaining &= remaining[candidate != guess]

        cs = [candidate.str[i] for i in range(n)]

        for i in range(n):
            c = candidate.str[i]
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

        for i in range(n):

            green_mask = outcome_df.guess.str[i] == outcome_df.candidate.str[i]

            yellow_mask = pd.Series(index=outcome_df.index, data=False)
            for ii in range(n):
                yellow_mask |= outcome_df.guess.str[i] == outcome_df.candidate.str[ii]
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

    def evaluate_guesses(self) -> pd.Series:

        outcome_df = Solver.outcome_after_guess(
            candidate=self.candidate_weight.index.to_series().rename("candidate"),
            guess=self.allowed_guess,
        )

        unique_candidate_and_outcome_counts_df = (
            outcome_df.groupby(["guess", "outcome"])
            .size()
            .rename("count")
            .reset_index()
        )

        c_df = self.candidate_weight.reset_index().rename(
            columns={"index": "candidate"}
        )

        weighted_outcome_df = unique_candidate_and_outcome_counts_df.merge(
            c_df, how="cross"
        )

        weighted_outcome_df["remaining"] = Solver.remaining(
            candidate=weighted_outcome_df.candidate,
            outcome=weighted_outcome_df.outcome,
            guess=weighted_outcome_df.guess,
        )

        weighted_outcome_df["guess_score_contribution"] = (
            weighted_outcome_df["count"]
            * weighted_outcome_df.weight
            * (1 - weighted_outcome_df.remaining.astype(float))
        )

        return (
            weighted_outcome_df.groupby("guess")
            .guess_score_contribution.sum()
            .rename("score")
        )

    def best_guesses(self) -> pd.DataFrame:
        guess_scores_df = self.evaluate_guesses().reset_index()
        best_guess_score = guess_scores_df.score.max()
        return guess_scores_df[guess_scores_df.score == best_guess_score]

    def a_best_guess(self) -> str:
        if self.candidate_weight.index.size == 1:
            return self.candidate_weight.index.values[0]
        return self.best_guesses().sample().guess.values[0]


def main():

    c_weights = candidate_weights()

    def _pre_display(guess: str, outcome: str):
        print(f"guess:'{guess}', outcome:{outcome}", end="")

    def _post_display(s:Solver):
        print(f", remaining candidates:{s.candidate_weight.index.size}")

    for solution in answers():

        first_guess = "tares"
        first_outcome = Solver.scalar_outcome_after_guess(
            candidate=solution, guess=first_guess
        )

        _pre_display(guess=first_guess, outcome=first_outcome)
        s = Solver.from_weights_and_guesses(
            candidate_weights=c_weights, allowed_guesses=c_weights.keys()
        ).with_guess(
            guess=first_guess,
            outcome=Solver.scalar_outcome_after_guess(
                candidate=solution, guess=first_guess
            ),
        )
        _post_display(s=s)

        while True:
            best_guess = s.a_best_guess()

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
