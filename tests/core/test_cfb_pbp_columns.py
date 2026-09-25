"""The CFB play-by-play is read as a column projection, so every raw column
the code touches must be in the projection.

The weekly CFB job loaded all ~300 columns of the .rds and peaked at 1.1 GB,
over Render's 512 MB cron limit; from 2026-09-13 it was killed before writing
ratings and the CFB board refused every game as stale. The parquet loader
reads PBP_COLUMNS only (190 MB, identical ratings). A new read of a column
outside the list would fail only on a live run -- this fails it here.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ingest import cfb_pbp as C  # noqa: E402


def test_every_raw_column_read_is_in_the_projection():
    src = (ROOT / "ingest" / "cfb_pbp.py").read_text()
    read = set(re.findall(r'(?:\braw|last_play|group|raw_with_scores)\["([A-Za-z_0-9]+)"\]', src))
    assert read, "the scan found nothing -- it is broken"
    missing = sorted(read - set(C.PBP_COLUMNS))
    assert not missing, f"read but not loaded: {missing}"


def test_the_season_is_loaded_from_parquet_not_the_full_rds():
    import inspect
    src = inspect.getsource(C.load_cfb_season)
    assert "_download_parquet" in src and "_download_and_parse_rds" not in src
    assert "columns=PBP_COLUMNS" in inspect.getsource(C._download_parquet)
