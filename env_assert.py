"""env_assert.py — the boot-time python-env assertion (F3 D3-06).

The sibling law of melite-audio-nodes/env_assert.py (audit 016, C1 —
closed there 2026-10-24, never applied to this pack). The burn
(2026-11-13 block, FRICTION_3 D3-06): the motion lane died MID-RUN
with ``TypeError: create_causal_mask() got an unexpected keyword
argument 'input_embeds'. Did you mean 'inputs_embeds'?`` — the pack's
KimodoTextToPose call drifted from the shared env's transformers
signature. The drift itself is cured in-tree (pack commits
ef0b656 → e7e0ffd → 1e7848f; ``input_embeds`` is gone from the pack),
but the TRAP stays: requirements.txt left transformers unpinned
("already ships in the base env"), Manager re-runs requirements only
on this pack's install/update, the venv is shared and last installer
wins — the next upstream rename lands silently and dies mid-run
again.

The law here (the audio precedent, verbatim in shape): the
RUN-BREAKING bound gets asserted at boot, in the pack's import path,
in install.py's own refusal grammar. The exact pin (==5.20.2) stays
at the install doors — two seats, two purposes: install converges the
reproducible version, boot refuses the run-breaking one. ComfyUI and
Manager surface failed pack imports at boot, so the maker sees the
named remedy in the console instead of a dead lane on press.
"""
from __future__ import annotations

import importlib.metadata

# The tested bound: the estate venv standard the pack's calls are
# written against (the 5.20 line — create_causal_mask's
# inputs_embeds-era signatures). Below the floor the legacy kwargs
# return; at 6+ the next rename lands — both are the kwarg-drift
# class this assert refuses at boot, never mid-run.
TRANSFORMERS_FLOOR = "5.20.2"
TRANSFORMERS_CEILING = "6"

REMEDY = (
    "melite-kimodo-nodes: the shared python env holds transformers {have} "
    "but the kimodo motion lane requires >={floor},<{ceil} (the tested "
    "bound — outside it the pack's KimodoTextToPose calls drift into the "
    "kwarg class that killed the lane: \"create_causal_mask() got an "
    "unexpected keyword argument 'input_embeds'\"). Another pack or the "
    "engine base likely re-pinned it after this pack installed. Remedy: "
    "pip install 'transformers>={floor},<{ceil}' (or ComfyUI-Manager → "
    "Update this pack, which re-runs its requirements)."
)


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".") if part.isdigit())


def assert_transformers_bound() -> None:
    """Refuse the pack import unless transformers meets the tested bound.

    @raises RuntimeError with the named remedy when the env un-converged.
    """
    have = "?"
    try:
        have = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeError(
            REMEDY.format(have="nothing (not installed)", floor=TRANSFORMERS_FLOOR, ceil=TRANSFORMERS_CEILING)
        ) from None
    if not (_version_tuple(have) >= _version_tuple(TRANSFORMERS_FLOOR) and _version_tuple(have) < _version_tuple(TRANSFORMERS_CEILING)):
        raise RuntimeError(
            REMEDY.format(have=have, floor=TRANSFORMERS_FLOOR, ceil=TRANSFORMERS_CEILING)
        )
