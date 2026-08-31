"""
In-script validation — Section 9.2's third detection layer.

Checks that fire BEFORE a git commit, catching bad data before it
ever reaches the static site, not just after the job "succeeds."
"""

import pandas as pd


class ValidationError(Exception):
    pass


def validate_pbp_data(df: pd.DataFrame, plays_per_week_min: int = 1000, plays_per_week_max: int = 2500) -> None:
    """
    Play count sanity check, normalized per week — the caller may pass
    a single week or a full season-to-date, so this checks plays-per-
    week-represented rather than assuming a fixed total.
    """
    n = len(df)
    n_weeks = df["week"].nunique() if "week" in df.columns and len(df) > 0 else 1
    n_weeks = max(n_weeks, 1)
    per_week = n / n_weeks

    if per_week < plays_per_week_min:
        raise ValidationError(f"Suspiciously few plays/week: {per_week:.0f} across {n_weeks} week(s) (expected at least {plays_per_week_min})")
    if per_week > plays_per_week_max:
        raise ValidationError(f"Suspiciously many plays/week: {per_week:.0f} across {n_weeks} week(s) (expected at most {plays_per_week_max})")


def validate_ratings(ratings: pd.DataFrame, expected_teams: int = 32) -> None:
    """Sanity-check the output ratings table itself before committing it."""
    if len(ratings) != expected_teams:
        raise ValidationError(f"Expected {expected_teams} teams in ratings, got {len(ratings)}")

    if ratings["total_rating"].isna().any():
        bad_teams = ratings[ratings["total_rating"].isna()].index.tolist()
        raise ValidationError(f"NaN rating(s) for: {bad_teams}")

    # A team rating swinging wildly outside a plausible range suggests a
    # pipeline bug (e.g. a bad opponent-adjustment iteration), not real
    # team quality — real DVOA-style ratings rarely exceed +/-60%.
    extreme = ratings[ratings["total_rating"].abs() > 0.60]
    if not extreme.empty:
        raise ValidationError(f"Implausibly extreme rating(s): {extreme.to_dict()}")


def validate_git_push_succeeded(push_result_returncode: int, push_stderr: str) -> None:
    if push_result_returncode != 0:
        raise ValidationError(f"git push failed (exit {push_result_returncode}): {push_stderr}")


if __name__ == "__main__":
    # Synthetic tests since this module has no real data dependency.
    import numpy as np

    good_ratings = pd.DataFrame(
        {"total_rating": np.random.uniform(-0.4, 0.4, 32)},
        index=[f"TEAM{i}" for i in range(32)],
    )
    print("Testing validate_ratings with good synthetic data (should pass silently)...")
    validate_ratings(good_ratings)
    print("  OK")

    print("Testing validate_ratings with a bad team count (should raise)...")
    try:
        validate_ratings(good_ratings.iloc[:30])
        print("  FAILED TO CATCH — bug in validation logic")
    except ValidationError as e:
        print(f"  correctly caught: {e}")

    print("Testing validate_ratings with an extreme value (should raise)...")
    bad_ratings = good_ratings.copy()
    bad_ratings.iloc[0, 0] = 5.0
    try:
        validate_ratings(bad_ratings)
        print("  FAILED TO CATCH — bug in validation logic")
    except ValidationError as e:
        print(f"  correctly caught: {e}")
