import pandas as pd


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
        outcome_df.loc[black_mask, "outcome"] += "⬛"

    return outcome_df


def scalar_outcome_after_guess(candidate: str, guess: str) -> str:
    single_candidate = pd.Series(data=[candidate], name="candidate")
    single_guess = pd.Series(data=[guess], name="guess")
    outcome_df = outcome_after_guess(candidate=single_candidate, guess=single_guess)
    return outcome_df.outcome.values[0]


def remaining(candidate: pd.Series, guess: pd.Series, outcome: pd.Series) -> pd.Series:
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
        black_mask = o == "⬛"

        remaining[green_mask] &= (c == g)[green_mask]

        has_g = pd.Series(index=candidate.index, data=False)
        for ii in range(n):
            has_g |= cs[ii] == g

        remaining[yellow_mask] &= has_g[yellow_mask]

        remaining[black_mask] &= ~has_g[black_mask]

    return remaining
