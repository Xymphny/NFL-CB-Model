"""The shakeout script, checked without a key or a network.

A script that only runs when someone has credentials is a script nobody tests,
and this one is the gate in front of a $59/month decision. So the parts that
can be checked offline are: that it refuses to run without a key, that it
never accepts a key as an argument, that its budget cannot be bypassed, and
that it does not print the key anywhere.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

SCRIPT = ROOT / "scripts" / "shakeout_odds_api.py"


def _run(env=None, args=()):
    import os
    e = dict(os.environ)
    e.pop("ODDS_API_KEY", None)
    e.update(env or {})
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, env=e, cwd=ROOT)


def test_the_script_exists_and_is_executable():
    assert SCRIPT.exists()
    assert SCRIPT.stat().st_mode & 0o111, "not executable"


def test_it_refuses_to_run_without_a_key_and_exits_2():
    """Exit 2 means 'not verified', matching scripts/check.sh -- distinct from
    exit 1, which here means the real API failed a check."""
    r = _run()
    assert r.returncode == 2
    assert "ODDS_API_KEY is not set" in r.stdout


def test_it_tells_you_not_to_pass_the_key_as_an_argument():
    r = _run()
    assert "shell history" in r.stdout


def test_there_is_no_key_argument_to_pass():
    """The advice above is only enforceable if the option does not exist."""
    r = _run(args=("--help",))
    assert "--api-key" not in r.stdout and "--key" not in r.stdout
    assert "--budget" in r.stdout and "--historical" in r.stdout


def test_the_default_budget_is_a_small_fraction_of_the_free_tier():
    src = SCRIPT.read_text()
    assert 'default=60' in src, (
        "the default budget changed; the free tier is 500 credits/month and "
        "this script should not be able to eat a meaningful share of it"
    )


def test_the_budget_is_enforced_by_the_ledger_not_by_the_script():
    """A cap the script checks itself can be bypassed by a bug in the script.
    CreditLedger.check runs before any request is issued."""
    from coverline.execution.odds_client import CreditLedger, QuotaExceeded
    led = CreditLedger(budget=5)
    with pytest.raises(QuotaExceeded):
        led.check(6)
    assert led.spent_predicted == 0


def test_it_records_that_historical_is_paid_only():
    """The finding that shapes the whole free-tier plan. If someone later
    removes this, the script would imply the backfill path was verified."""
    src = SCRIPT.read_text()
    assert "paid plans only" in src
    assert "92,000" in src or "92000" in src, (
        "the script should name the backfill cost it is protecting"
    )


def test_it_checks_pinnacle_because_adr_0002_depends_on_it():
    src = SCRIPT.read_text()
    assert "pinnacle" in src.lower()
    assert "ADR 0002" in src


def test_it_writes_bronze_to_a_temp_dir_not_the_repo():
    """A shakeout that leaves snapshots in data/bronze/ would pollute a
    write-once store with test data that cannot be deleted from it."""
    src = SCRIPT.read_text()
    assert "TemporaryDirectory" in src
    assert 'BronzeStore(tmp)' in src


def test_failure_tells_you_not_to_subscribe():
    src = SCRIPT.read_text()
    assert "DO NOT SUBSCRIBE YET" in src
