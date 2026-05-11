from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

# =====================================================
# =========       Constants and options       =========
# =====================================================

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "saved" / "log"
DEFAULT_GLOB = "**/info.log"

TS_LINE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) - (train|trainer) - INFO - "
)
BACKFILL_LINE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - train - INFO - "
    r"(Training started at|Training finished at|Total training time:)"
)


# =====================================================
# =========           Functions               =========
# =====================================================

def parse_ts(head: str, ms: str) -> datetime:
    """Parse timestamp string + milliseconds into datetime."""
    base = datetime.strptime(head, "%Y-%m-%d %H:%M:%S")
    return base.replace(microsecond=int(ms) * 1000)


def strip_backfill(lines: list[str]) -> list[str]:
    """Remove any previously inserted backfill lines."""
    return [ln for ln in lines if not BACKFILL_LINE.match(ln)]


def analyze_lines(
    lines: list[str],
) -> tuple[datetime, datetime, str, str, int] | None:
    """
    Return (start_dt, end_dt, start_asctime, end_asctime, last_train_line_idx).
    Insert "Training started" after last_train_line_idx.
    """
    first_trainer_idx = None
    last_train_idx = None
    last_train_before = None
    last_ts_end = None

    for i, line in enumerate(lines):
        m = TS_LINE.match(line)
        if not m:
            continue
        head, ms, logger = m.groups()
        ts = parse_ts(head, ms)
        full_asctime = f"{head},{ms}"
        last_ts_end = (ts, full_asctime, i)

        if logger == "trainer" and first_trainer_idx is None:
            first_trainer_idx = i

        if logger == "train" and first_trainer_idx is None:
            last_train_idx = i
            last_train_before = (ts, full_asctime)

    if (
        first_trainer_idx is None
        or last_train_idx is None
        or last_train_before is None
        or last_ts_end is None
    ):
        return None

    start_dt, start_asctime = last_train_before
    end_dt, end_asctime, _end_idx = last_ts_end
    if end_dt < start_dt:
        return None
    return start_dt, end_dt, start_asctime, end_asctime, last_train_idx


def process_file(path: Path, dry_run: bool, max_seconds: float | None) -> str:
    """Insert training start/finish timestamps into a single log file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = strip_backfill(text.splitlines())
    parsed = analyze_lines(lines)
    if parsed is None:
        return "skip(no_complete_training)"

    start_dt, end_dt, start_ac, end_ac, last_train_idx = parsed
    elapsed = end_dt - start_dt
    if max_seconds is not None and elapsed.total_seconds() > max_seconds:
        return f"skip(duration>{max_seconds}s_suspicious)"

    msg_start = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    msg_end = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    start_line = f"{start_ac} - train - INFO - Training started at {msg_start}"
    end_lines = [
        f"{end_ac} - train - INFO - Training finished at {msg_end}",
        f"{end_ac} - train - INFO - Total training time: {elapsed.total_seconds():.2f}s ({elapsed})",
    ]

    if dry_run:
        return f"would_write duration={elapsed}"

    body = lines[: last_train_idx + 1] + [start_line] + lines[last_train_idx + 1 :]
    final_lines = body + end_lines
    out = "\n".join(final_lines)
    if text.endswith("\n") or not text:
        out += "\n"
    path.write_text(out, encoding="utf-8")
    return f"wrote duration={elapsed}"


# =====================================================
# =========              Main                 =========
# =====================================================

def main():
    """
    Insert "Training started" (before epoch logs) and append "Training finished" / duration
    to old info.log files — same messages as train.py, in chronological order.

    Removes prior backfill lines if present so you can re-run to repair placement.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument(
        "--max-hours", type=float, default=None,
        help="Skip if inferred duration exceeds this (hours).",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root: Path = args.root
    if not root.is_dir():
        print(f"Missing log root: {root}")
        return

    max_sec = args.max_hours * 3600 if args.max_hours is not None else None
    paths = sorted(root.glob(args.glob))
    for p in paths:
        status = process_file(p, args.dry_run, max_sec)
        print(f"{p.relative_to(root.parent.parent)}: {status}")


if __name__ == "__main__":
    main()
