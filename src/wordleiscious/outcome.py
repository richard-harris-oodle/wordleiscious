import string

import pandas as pd
from colorama import Back, Fore, Style


def outcome_after_guess(candidate: pd.Series, guess: pd.Series) -> pd.DataFrame:
    n = candidate.str.len().max()

    outcome_df = pd.merge(candidate, guess, how="cross")

    outcome_df["outcome"] = ""

    cs = [outcome_df.candidate.str[i] for i in range(n)]
    gs = [outcome_df.guess.str[i] for i in range(n)]

    for i in range(n):

        g = gs[i]
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

    os = [outcome_df.outcome.str[i] for i in range(n)]

    # Turn 🟨 to ⬛ if all instances of the character are accounted for in 🟩.
    for i in range(n):
        g = gs[i]
        o = os[i]

        g_is_green_count = pd.Series(index=outcome_df.index, data=0)
        g_count = pd.Series(index=outcome_df.index, data=0)

        for ii in range(n):
            oo = os[ii]
            gg = gs[ii]

            g_is_green_count[(oo == "🟩") & (gg == g)] += 1
            g_count[gg == g] += 1

        yellow_to_black_mask = (o == "🟨") & g_is_green_count == g_count
        outcome_df.outcome[yellow_to_black_mask] = (
            outcome_df.outcome[yellow_to_black_mask].str[:i]
            + "⬛"
            + outcome_df.outcome[yellow_to_black_mask].str[i + 1 :]
        )

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

        remaining[black_mask] &= (same_g_count == 0)[black_mask]

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
