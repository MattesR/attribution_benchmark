"""Split long records into windows that the annotators can read in full.

Comma already cuts long texts into records of at most 128,000 characters
(Gutenberg: blind cuts, often mid-word; DOAB: mostly whole sections). Those
records are the unit the model was trained on, but they are longer than
propella (50k chars) and WebOrganizer (8,192 tokens) read. --windows N splits
every record longer than N characters into windows of at most N characters:

- the number of windows is fixed per record, ceil(len / (0.9 * N)), and they
  are of similar size (a 33k record becomes two of ~16.5k, not 32k + 1k). The
  10% slack lets each cut move to a good boundary without forcing an extra
  window (Gutenberg records are almost exactly 4 x 32k);
- each cut is at the best boundary near the target: a heading line (Markdown
  heading or CHAPTER), else a paragraph break, else a line break, else a
  sentence end; only if none exists, a hard cut;
- windows are exact, contiguous character ranges [start, end) of the record,
  so (shard, row, start, end) maps back to the raw text.
"""

from __future__ import annotations

import math
import re

# Boundary patterns, best first. A cut is placed at the match's end, i.e. the
# next window starts with the heading / paragraph / line / sentence.
BOUNDARIES = [
    re.compile(r"\n(?=#{1,3} |\*{0,2}(CHAPTER|Chapter) [IVXLCDM\d]+)"),
    re.compile(r"\n\s*\n"),
    re.compile(r"\n"),
    re.compile(r"[.!?][\"”’)\]]?\s"),
]


def split_windows(text: str, max_chars: int) -> list[tuple[int, int]]:
    if len(text) <= max_chars:
        return [(0, len(text))]
    n = math.ceil(len(text) / (0.9 * max_chars))
    windows, pos = [], 0
    for k in range(n - 1):
        target = (len(text) - pos) / (n - k)  # equal share of what is left
        # late enough that the rest still fits into the remaining windows
        lo = max(pos + int(0.6 * target), len(text) - (n - k - 1) * max_chars)
        hi = pos + min(max_chars, int(1.15 * target))
        cut = None
        for rx in BOUNDARIES:
            matches = [m.end() for m in rx.finditer(text, lo, hi)]
            if matches:
                # the boundary closest to the target, so windows stay balanced
                cut = min(matches, key=lambda c: abs(c - (pos + target)))
                break
        cut = cut or hi
        windows.append((pos, cut))
        pos = cut
    windows.append((pos, len(text)))
    return windows
