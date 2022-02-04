import pandas as pd

from wordleiscious.words import candidate_weights
from outcome import outcome_after_guess, remaining
from tqdm.auto import tqdm
import swifter
import numpy as np

tqdm.pandas()


def main():

    word_weights_df = (
        pd.DataFrame.from_dict(candidate_weights(), orient="index")
        .reset_index()
        .rename(columns={"index": "word", 0: "weight"})
        .sample(n=100)
        .reset_index(drop=True)
    )

    candidate_weights_df = word_weights_df.rename(columns={"word": "candidate"})
    guess = candidate_weights_df.candidate.rename("guess")
    other_candidate_weights_df = word_weights_df.rename(
        columns={"word": "other_candidate"}
    )

    def find_impacted_candidates(c):

        if not isinstance(c, str):
            return

        outcome_df = outcome_after_guess(
            candidate=pd.Series(data=c, name="candidate"),
            guess=guess,
        )

        remaing_df = outcome_df.merge(other_candidate_weights_df, how="cross")

        remaing_df["eliminated"] = ~remaining(
            candidate=remaing_df.other_candidate,
            outcome=remaing_df.outcome,
            guess=remaing_df.guess,
        )

        return outcome_df.outcome.value_counts()

    _find_impacted_candidates = np.vectorize(find_impacted_candidates)

    # outcomes_df = outcome_after_guess(
    #     candidate=c,
    #     guess=guess,
    # )
    # return outcomes_df.outcome

    os = candidate_weights_df.candidate.swifter.apply(find_impacted_candidates)

    pass

    # other_candidates = candidate_weights_df.word.rename("other_candidate")
    #
    # def _(guess_df: pd.DataFrame):
    #
    #     unique_outcome_df = guess_df.groupby("outcome").candidate.unique()
    #
    #     df = other_candidates.reset_index().merge(unique_outcome_df, how="cross")
    #
    #     r = remaining(candidate=df.other_candidate, guess=df.guess, outcome=df.outcome)
    #
    #     surviving = r.mean()
    #
    #     remaining_df = df[r]
    #
    #     return None
    #
    # outcomes = all_outcomes_df.groupby("guess").apply(_)

    pass


if __name__ == "__main__":
    main()
