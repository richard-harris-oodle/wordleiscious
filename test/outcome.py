import pandas as pd
import pytest
from wordleiscious.outcome import (
    scalar_outcome_after_guess,
    remaining,
    outcome,
)


@pytest.mark.parametrize(
    argnames=["guess", "candidate", "expected_outcome"],
    argvalues=[
        ("tares", "stare", "🟨🟨🟨🟨🟨"),
        ("black", "lacks", "⬛🟨🟨🟨🟨"),
        ("lacks", "black", "🟨🟨🟨🟨⬛"),
        ("fills", "zills", "⬛🟩🟩🟩🟩"),
        ("field", "fills", "🟩🟩⬛🟩⬛"),
        ("loser", "fills", "🟨⬛🟨⬛⬛"),
        ("lills", "fills", "⬛🟩🟩🟩🟩"),
    ],
)
def test_scalar_outcome(guess: str, candidate: str, expected_outcome: str):
    outcome = scalar_outcome_after_guess(candidate=candidate, guess=guess)
    assert outcome == expected_outcome


@pytest.mark.parametrize(
    argnames=["candidate_and_guess_df", "expected_outcome"],
    argvalues=[
        (
            pd.DataFrame(
                {
                    "guess": [
                        "tares",
                        "black",
                        "lacks",
                        "fills",
                        "field",
                        "loser",
                        "lills",
                    ],
                    "candidate": [
                        "stare",
                        "lacks",
                        "black",
                        "zills",
                        "fills",
                        "fills",
                        "fills",
                    ],
                }
            ),
            pd.Series(
                name="outcome",
                data=[
                    "🟨🟨🟨🟨🟨",
                    "⬛🟨🟨🟨🟨",
                    "🟨🟨🟨🟨⬛",
                    "⬛🟩🟩🟩🟩",
                    "🟩🟩⬛🟩⬛",
                    "🟨⬛🟨⬛⬛",
                    "⬛🟩🟩🟩🟩",
                ],
            ),
        )
    ],
)
def test_outcome(candidate_and_guess_df, expected_outcome: pd.Series):
    outcome_df = outcome(candidate_and_guess_df=candidate_and_guess_df)
    pd.testing.assert_series_equal(outcome_df.outcome, expected_outcome)


@pytest.mark.parametrize(
    argnames=["candidate", "guess", "outcome", "expected_remaining"],
    argvalues=[
        (
            pd.Series(
                name="candidate",
                data=[
                    "stare",
                    "lacks",
                    "black",
                    "zills",
                    "fills",
                    "fills",
                    "fills",
                    "fills",
                    "stare",
                    "lacks",
                    "black",
                    "zills",
                    "fills",
                    "fills",
                ],
            ),
            pd.Series(
                name="guess",
                data=[
                    "tares",
                    "black",
                    "lacks",
                    "fills",
                    "field",
                    "loser",
                    "lills",
                    "tares",
                    "black",
                    "lacks",
                    "fills",
                    "field",
                    "loser",
                    "lills",
                ],
            ),
            pd.Series(
                name="outcome",
                data=[
                    "🟨🟨🟨🟨🟨",
                    "⬛🟨🟨🟨🟨",
                    "🟨🟨🟨🟨⬛",
                    "⬛🟩🟩🟩🟩",
                    "🟩🟩⬛🟩⬛",
                    "🟨⬛🟨⬛⬛",
                    "⬛🟩🟩🟩🟩",
                    "🟨🟨🟨🟨🟨",
                    "⬛🟨🟨🟨🟨",
                    "🟨🟨🟨🟨⬛",
                    "⬛🟩🟩🟩🟩",
                    "🟩🟩⬛🟩⬛",
                    "🟨⬛🟨⬛⬛",
                    "⬛🟩🟩🟩🟩",
                ],
            ),
            pd.Series(
                name="remaining",
                data=[
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                ],
            ),
        )
    ],
)
def test_remaining(
    candidate: pd.Series,
    guess: pd.Series,
    outcome: pd.Series,
    expected_remaining: pd.Series,
):
    actual_remaining = remaining(candidate=candidate, guess=guess, outcome=outcome)
    pd.testing.assert_series_equal(actual_remaining, expected_remaining)
