"""Pure parts of the evaluation script (no model needed)."""
import pytest

pytest.importorskip("torch")
from src.tasks.splits import HOLDOUT
from src.training import evaluate as E


def test_wilson_interval_is_sane():
    assert E.wilson(0, 0) == (0.0, 0.0)
    lo, hi = E.wilson(0, 4)
    assert lo == 0.0 and 0.3 < hi < 0.6                    # 0/4 does not mean "exactly 0%"
    lo, hi = E.wilson(2, 4)
    assert lo < 0.5 < hi
    lo, hi = E.wilson(40, 40)
    assert hi == 1.0 and lo > 0.85
    assert E.wilson(5, 20)[1] - E.wilson(5, 20)[0] < E.wilson(1, 4)[1] - E.wilson(1, 4)[0]


def test_summarize_splits_trained_on_from_held_out_and_ignores_the_rest():
    held = sorted(HOLDOUT)[0]
    rows = ([{"template": "a", "reward": 1.0}] * 3 + [{"template": "a", "reward": 0.0}]
            + [{"template": held, "reward": 0.0}] * 4 + [{"template": "other", "reward": 1.0}] * 2)
    s = E.summarize(rows, {"a"})
    assert (s["trained-on"]["passed"], s["trained-on"]["n"]) == (3, 4)
    assert (s["held-out"]["passed"], s["held-out"]["n"]) == (0, 4)
    assert s["templates"]["other"] == (2, 2)               # recorded, but in neither headline group
    md = E.render("t", s, {"a"})
    assert "| trained-on | 3 | 4 | 75.0% |" in md and "| held-out | 0 | 4 | 0.0% |" in md
    assert f"| {held} | held-out | 0/4 |" in md
