import pandas as pd
import pytest
from wordleiscious.outcome import (
    scalar_outcome_after_guess,
    outcome_after_guess,
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
