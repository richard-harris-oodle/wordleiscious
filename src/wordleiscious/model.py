import numpy as np
import pandas as pd
from tqdm import tqdm

from wordleiscious.words import all_words, weights


class Solver:
    def __init__(self, weighted_candidate_df: pd.DataFrame, allowed_guess: pd.Series):
        self.weighted_candidate_df = weighted_candidate_df
        self.allowed_guess = allowed_guess

    def with_guess(self, guess: str, outcome: str) -> "Solver":
        remaining = Solver.remaining(
            candidate=self.weighted_candidate_df.index.to_series(),
            guess=guess,
            outcome=outcome,
        )
        weighted_candidate_df = self.weighted_candidate_df[remaining].copy()
        allowed_guess = self.allowed_guess[self.allowed_guess != guess].copy()
        return Solver(
            weighted_candidate_df=weighted_candidate_df, allowed_guess=allowed_guess
        )

    @staticmethod
    def remaining(candidate: pd.Series, guess: str, outcome: str) -> pd.Series:
        n = len(outcome)
        assert len(guess) == n, f"len(guess)({len(guess)}) != len(outcome)({n})"

        remaining = pd.Series(index=candidate.index, name="remaining", data=True)

        remaining &= remaining[candidate != guess]

        for i, (g, o) in enumerate(zip(guess, outcome)):
            if o == "🟩":
                remaining &= candidate.str[i] == g
            elif o == "🟨":
                remaining &= candidate.str.contains(g)
            elif o == "🟦":
                remaining &= ~candidate.str.contains(g)

        return remaining

    @staticmethod
    def outcome_after_guess(solution: pd.Series, guess: str) -> pd.Series:

        outcome = pd.Series(name="outcome", index=solution.index, data="")

        for i, g in enumerate(guess):
            green_mask = solution.str[i] == g
            yellow_mask = ~green_mask & solution.str.contains(g)
            black_mask = ~green_mask & ~yellow_mask

            outcome[green_mask] += "🟩"
            outcome[yellow_mask] += "🟨"
            outcome[black_mask] += "🟦"

        return outcome

    @staticmethod
    def scalar_outcome_after_guess(solution: str, guess: str) -> str:
        solution_series = pd.Series(data=solution, name="candidate")
        outcome = Solver.outcome_after_guess(
            solution=pd.Series(data=solution_series, name="candidate"), guess=guess
        )
        return outcome.values[0]

    def evaluate_guess(self, guess: str) -> float:

        outcome = Solver.outcome_after_guess(
            solution=self.weighted_candidate_df.index.to_series(), guess=guess
        )

        outcome_counts = outcome.value_counts()

        unique_outcomes, weights = zip(*outcome_counts.iteritems())

        fractions_remaining = [
            (
                Solver.remaining(
                    candidate=self.weighted_candidate_df.index.to_series(),
                    guess=guess,
                    outcome=unique_outcome,
                )
                * self.weighted_candidate_df.frequency
            ).mean()
            for unique_outcome in unique_outcomes
        ]

        return np.average(a=fractions_remaining, weights=weights)

    def evaluate_guesses(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            {
                guess: {"fraction_remaining": self.evaluate_guess(guess=guess)}
                for guess in tqdm(self.allowed_guess, leave=False)
            },
            orient="index",
        )

    def best_guesses(self):
        guess_df = self.evaluate_guesses()
        best_guess_fraction = guess_df.fraction_remaining.min()
        return guess_df[guess_df.fraction_remaining == best_guess_fraction].copy()

    def a_best_guess(self) -> str:
        if self.weighted_candidate_df.index.size == 1:
            return self.weighted_candidate_df.index.values[0]
        return self.best_guesses().sample().index.values[0]


# def emojify(outcome: str):
#     return outcome.replace("g", "🟩").replace("y", "🟨").replace("b", "🟦")


def main2():

    ft_df = weights()
    candidate_index = pd.Index(name="word", data=list(all_words()))

    weighted_candidate_df = (
        ft_df.reindex(candidate_index)
        .fillna(ft_df.weight.min())
        .sort_values("weight", ascending=False)
    )

    weighted_candidate_df.weight /= weighted_candidate_df.weight.sum()

    solution = "perky"
    # first_guess = "raise"

    # outcome = Solver.scalar_outcome_after_guess(solution=solution, guess=first_guess)

    # print(f"guess:'{first_guess}', outcome:{outcome}", end="")
    # s = Solver.with_first_guess(
    #     weighted_candidate_df=weighted_candidate_df,
    #     allowed_guess=weighted_candidate_df.index.to_series(),
    #     guess=first_guess,
    #     outcome=outcome,
    # )
    # print(f", remaining candidates:{s.weighted_candidate_df.index.size}")

    s = Solver(
        weighted_candidate_df=weighted_candidate_df,
        allowed_guess=weighted_candidate_df.index.to_series(),
    )

    while True:
        best_guess = s.a_best_guess()

        best_outcome = Solver.scalar_outcome_after_guess(
            solution=solution, guess=best_guess
        )
        print(f"guess:'{best_guess}', outcome:{best_outcome}", end="")
        if best_outcome == "🟩🟩🟩🟩🟩":
            break
        s = s.with_guess(guess=best_guess, outcome=best_outcome)
        print(f", remaining candidates:{s.weighted_candidate_df.index.size}")


if __name__ == "__main__":
    main2()


# guess:'raise', outcome:🟨🟦🟦🟦🟨, remaining candidates:602
# guess:'donut', outcome:🟦🟦🟦🟦🟦, remaining candidates:128
# guess:'glyph', outcome:🟦🟦🟨🟨🟦, remaining candidates:10
# guess:'remex', outcome:🟨🟩🟦🟨🟦, remaining candidates:4
# guess:'kieve', outcome:🟨🟦🟨🟦🟨, remaining candidates:1
# guess:'perky', outcome:🟩🟩🟩🟩🟩
