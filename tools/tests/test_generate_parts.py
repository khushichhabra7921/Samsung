import csv

import pytest

from generate_outputs import assign_part
from handscribe_xr.contracts import ROOT


def epic_ids_by_length():
    with (ROOT / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["dataset"] == "epic_kitchens_100"]
    return [row["video_id"] for row in sorted(rows, key=lambda row: float(row["duration_sec"]))], {
        row["video_id"]: float(row["duration_sec"]) for row in rows}


@pytest.mark.parametrize("n", [2, 3])
def test_parts_cover_every_video_once_and_balance_length(n):
    order, seconds = epic_ids_by_length()
    parts = [assign_part(order, f"{k}/{n}") for k in range(1, n + 1)]
    assert sorted(v for part in parts for v in part) == sorted(order)
    totals = [sum(seconds[v] for v in part) for part in parts]
    assert max(totals) / min(totals) < 1.35


def test_bad_part_is_rejected():
    with pytest.raises(SystemExit):
        assign_part(["a"], "3/2")
