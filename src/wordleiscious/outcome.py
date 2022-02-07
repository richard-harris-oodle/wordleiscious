import pandas as pd
from colorama import Back, Fore, Style


def outcome_after_guess(candidate: pd.Series, guess: pd.Series) -> pd.DataFrame:
    candidate_and_guess_df = pd.merge(candidate, guess, how="cross")
    return outcome(candidate_and_guess_df=candidate_and_guess_df)


def outcome(candidate_and_guess_df: pd.DataFrame) -> pd.DataFrame:
    n = candidate_and_guess_df.candidate.str.len().max()

    outcome_df = candidate_and_guess_df.copy()

    outcome_df["outcome"] = ""

    guess_chars_df = pd.DataFrame.from_dict(
        {i: candidate_and_guess_df.guess.str[i] for i in range(n)}
    )

    candidate_chars_df = pd.DataFrame.from_dict(
        {i: candidate_and_guess_df.candidate.str[i] for i in range(n)}
    )

    same_char_count_df = pd.DataFrame(
        index=candidate_and_guess_df.index, columns=range(n), data=0
    )

    same_char_green_count_df = pd.DataFrame(
        index=candidate_and_guess_df.index, columns=range(n), data=0
    )

    for i in range(n):
        for ii in range(n):
            if i == ii:
                continue
            same_char = guess_chars_df[i] == candidate_chars_df[ii]
            same_char_count_df[i] += same_char
            same_char_green_count_df[i] += same_char & (
                guess_chars_df[ii] == candidate_chars_df[ii]
            )

    same_char_not_known_counts_df = same_char_count_df - same_char_green_count_df

    for i in range(n):

        green_mask = guess_chars_df[i] == candidate_chars_df[i]
        yellow_mask = ~green_mask & (same_char_not_known_counts_df[i] > 0)
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

    # remaining &= remaining[candidate != guess]

    cs = [candidate.str[i] for i in range(n)]
    os = [outcome.str[i] for i in range(n)]
    gs = [guess.str[i] for i in range(n)]

    for i in range(n):
        c = cs[i]
        o = os[i]
        g = gs[i]

        green_mask = o == "🟩"
        yellow_mask = o == "🟨"
        black_mask = o == "⬛"

        same_g_count = pd.Series(  # The number of times g occurs in the candidate
            index=candidate.index, data=0
        )
        same_g_accounted_for = pd.Series(  # The number of times g occurs and is 🟩 or 🟨
            index=candidate.index, data=0
        )

        for ii in range(n):
            same_g_count += cs[ii] == g
            same_g_accounted_for += (cs[ii] == g) & (os[ii] == "🟩")

        remaining[green_mask] &= (c == g)[green_mask]

        remaining[yellow_mask] &= (same_g_count > same_g_accounted_for)[yellow_mask]

        remaining[black_mask] &= (same_g_count == same_g_accounted_for)[black_mask]

    return remaining.astype(bool)


background_lookup = {
    "🟩": Back.GREEN,
    "🟨": Back.YELLOW,
    "⬛": Back.BLACK,
}


def display(guess: str, outcome: str) -> str:
    return "".join(
        f"{background_lookup[o]}{Fore.LIGHTWHITE_EX}{Style.BRIGHT} {g.upper()} {Style.RESET_ALL}"
        for g, o in zip(guess, outcome)
    )
