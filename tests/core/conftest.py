"""Suite-wide guards.

THE SITE FILES ARE THE CRONS', NOT THE TESTS'. data/site/*.json is what the
capture and live-inputs jobs publish. A test that runs an exporter against
the default output rewrote them in the working tree -- changes nobody made,
one `git commit -am` away from overwriting what the jobs publish. Every
exporter's output directory is pointed at a temp dir for every test, and
the session fails if any data/site file changed anyway.
"""

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "data" / "site"
for p in (ROOT / "scripts", ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

EXPORTERS = ("export_board", "export_record", "export_players")


def _digest() -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(SITE.glob("*.json"))}


def pytest_sessionstart(session):
    session.config._site_digest = _digest()


def pytest_sessionfinish(session, exitstatus):
    before = getattr(session.config, "_site_digest", None)
    if before is None:
        return
    after = _digest()
    changed = sorted(n for n in set(before) | set(after) if before.get(n) != after.get(n))
    if changed:
        print(f"\nTESTS WROTE data/site: {', '.join(changed)}. Restore with "
              "`git checkout -- data/site` and point the test at a temp dir.")
        session.exitstatus = 1


@pytest.fixture(autouse=True)
def _board_context_stays_offline(monkeypatch):
    """No test reaches ESPN, nflverse or Open-Meteo through the board's
    context; every source soft-fails, which is itself the path tested most."""
    monkeypatch.setenv("COINFLIP_CONTEXT_OFFLINE", "1")


@pytest.fixture(autouse=True)
def _site_exports_go_to_tmp(monkeypatch, tmp_path):
    import importlib
    for name in EXPORTERS:
        mod = sys.modules.get(name) or importlib.import_module(name)
        monkeypatch.setattr(mod, "SITE", tmp_path / "site", raising=False)
