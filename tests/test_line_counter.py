"""Small checks of the line-crossing logic with made-up tracks (no video needed).

Run:  python tests/test_line_counter.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from counter.detector import Track  # noqa: E402
from counter.line_counter import LineCounter  # noqa: E402

FPS = 25


def box_at(x, y, size=10):
    """A box whose centre is (x, y)."""
    return (x - size, y - size, x + size, y + size)


def run(path, counter=None, tid=1, cls="car"):
    """Move one object along a list of centre points, frame by frame."""
    lc = counter or LineCounter((0, 100), (200, 100))  # horizontal line at y=100, x from 0 to 200
    for i, (x, y) in enumerate(path):
        lc.update([Track(tid, cls, 0.9, box_at(x, y))], i, FPS)
    return lc


def test_moving_down_counts_in():
    lc = run([(50, 80), (50, 95), (50, 105), (50, 120)])
    assert lc.totals() == {"IN": 1, "OUT": 0}, lc.totals()


def test_moving_up_counts_out():
    lc = run([(50, 120), (50, 105), (50, 95), (50, 80)])
    assert lc.totals() == {"IN": 0, "OUT": 1}, lc.totals()


def test_never_counted_twice():
    # Crosses down, back up, down again: still only one count.
    lc = run([(50, 80), (50, 120), (50, 80), (50, 120)])
    assert len(lc.events) == 1, lc.events


def test_no_count_without_crossing():
    lc = run([(50, 80), (60, 85), (70, 90), (80, 95)])
    assert lc.events == []


def test_passing_beyond_line_end_not_counted():
    # Line ends at x=200; this object crosses y=100 at x=300.
    lc = run([(300, 80), (300, 120)])
    assert lc.events == []


def test_stopping_exactly_on_line():
    # Centre lands exactly on the line for a frame, then continues: count once.
    lc = run([(50, 90), (50, 100), (50, 110)])
    assert lc.totals() == {"IN": 1, "OUT": 0}, lc.totals()


def test_two_objects_counted_separately():
    lc = LineCounter((0, 100), (200, 100))
    for i, (y1, y2) in enumerate([(80, 120), (120, 80)]):
        lc.update([Track(1, "car", 0.9, box_at(50, y1)), Track(2, "truck", 0.9, box_at(150, y2))], i, FPS)
    assert lc.counts == {("car", "IN"): 1, ("truck", "OUT"): 1}, lc.counts


if __name__ == "__main__":
    tests = [f for name, f in dict(globals()).items() if name.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
