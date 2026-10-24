"""Placeholder vocabulary for fixture recipes.

Everything here is driven by an explicit `random.Random(seed)`, never the global
RNG, so a template plus a seed always produces byte-identical fixtures. That is
what makes a failing calibration run reproducible: "template X, seed 41" is a
complete bug report.

Stdlib only.
"""

from __future__ import annotations

import random
import re
from typing import Optional

# Short, lowercase, shell-safe. No characters that need quoting, nothing that
# collides with glob syntax, and no real filenames an agent might recognise and
# special-case.
WORDS = [
    "alfa", "bravo", "cedar", "delta", "ember", "flint", "gamma", "hazel",
    "indigo", "jade", "kelp", "lumen", "maple", "nimbus", "onyx", "pearl",
    "quartz", "rowan", "slate", "topaz", "umber", "violet", "willow", "xenon",
    "yarrow", "zephyr", "amber", "basil", "cobalt", "dune", "echo", "fern",
    "grove", "heath", "ivory", "juniper", "kite", "larch", "mica", "nectar",
    "ochre", "pine", "quill", "reed", "sage", "thistle", "umbra", "vetch",
    "wren", "yucca", "zinc", "almond", "birch", "cinder", "dusk", "elm",
    "frost", "glade", "hollow", "iris",
]

_PLACEHOLDER = re.compile(r"\{(word|int|hex)\}")


class WordPool:
    """Draws words without replacement, so generated filenames never collide.

    Collisions would silently shrink a fixture: two rules rendering to the same
    name means `count: [3, 3]` produces two files, and the expectations derived
    from the tag would then disagree with what is on disk.
    """

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._pool = WORDS[:]
        rng.shuffle(self._pool)
        self._suffix = 0

    def take(self) -> str:
        if self._pool:
            return self._pool.pop()
        # Exhausted the vocabulary: fall back to suffixed words, still unique.
        self._suffix += 1
        return f"{self._rng.choice(WORDS)}{self._suffix}"


def render(pattern: str, pool: WordPool, rng: random.Random,
           ints: Optional[list[int]] = None) -> str:
    """Substitute {word}, {int} and {hex} placeholders in a pattern.

    {word} is drawn from the pool (unique); {int} and {hex} are not, because
    duplicate *content* is harmless -- only names must be distinct.

    Pass `ints` to collect the integers generated, which is what lets a template
    assert an answer computed from file *contents* (a sum, a maximum) rather than
    only from filenames and counts. The RNG is consumed identically either way,
    so collecting never changes the fixture.
    """
    def sub(m: re.Match) -> str:
        kind = m.group(1)
        if kind == "word":
            return pool.take()
        if kind == "int":
            value = rng.randint(1000, 9999)
            if ints is not None:
                ints.append(value)
            return str(value)
        return f"{rng.randrange(16**6):06x}"

    return _PLACEHOLDER.sub(sub, pattern)
