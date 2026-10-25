"""Which templates are held out of training entirely.

Evaluating on held-out SEEDS of the templates we trained on measures how well the policy
handles new fixtures. It says nothing about new KINDS of task, and it is easy to mistake
for generalisation. These eight, one per skill family, are never used for SFT or RL by
any code path: the training entry point builds its rollout from the other templates
only, so a held-out instruction reaching it raises KeyError instead of being trained on.
"""

HOLDOUT = (
    "count-nested",        # counting
    "list-hidden",         # listing, dotfiles
    "replace-in-place",    # sed -i
    "delete-large",        # find -size
    "remove-world-write",  # symbolic chmod
    "concat-parts",        # cat
    "dedupe-report",       # grep over contents
    "highest-port",        # aggregating numbers
)


def split(templates):
    """(train, heldout) lists. Fails loudly if a held-out id is not a real template,
    because a typo here would silently put a held-out task into training."""
    ids = {t.id for t in templates}
    missing = [h for h in HOLDOUT if h not in ids]
    if missing:
        raise ValueError(f"held-out templates do not exist: {missing}")
    held = set(HOLDOUT)
    return [t for t in templates if t.id not in held], [t for t in templates if t.id in held]
