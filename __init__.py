"""Tech Noir Kimodo custom nodes for ComfyUI.

Wraps NVIDIA Kimodo — kinematic motion diffusion model for text-to-3D
human/humanoid motion. Replaces the now-removed kimodo pool sidecar
(Originally media/poser/app_kimodo.py + media/motion/kimodo_pose.py).

Two nodes:
  KimodoLoader    — loads the diffusion model into VRAM (~17GB)
  KimodoTextToPose — runs text-to-pose inference, saves NPZ, returns result

KIMODO SOURCE RESOLUTION (2026-08-15, "No module named 'kimodo'";
converged 2026-11-12, audit 019 C6 — the cold-clone pack law): the
pack ships ONE install door, the pip leg in requirements.txt
(``kimodo @ git+https://github.com/nv-tlabs/kimodo.git@54257dd``).
The repo-layout sys.path bootstrap that used to live here (vendor/
kimodo through the custom_nodes symlink, then the retired /opt/kimodo
image layout) is DELETED: it could only ever fire on a dev box, and a
cold Manager install rode the pip leg anyway. nodes.py imports kimodo
lazily (``kimodo.model.loading`` / ``kimodo.model.load_model`` inside
KimodoLoader.load and its per-load posture helper),
so the pack registers fine either way; an unimportable kimodo lands
at KimodoLoader.load with the breadcrumb below in the log.

requirements.txt declares the dependencies beyond the image base env
(hydra-core — kimodo's config instantiate(); the kimodo pip leg;
anny + multihmr2 + trimesh + xatlas for the folded-in estimator).
"""
from __future__ import annotations

import logging

log = logging.getLogger("melite-kimodo")

try:
    import kimodo  # noqa: F401 — the requirements.txt pip leg owns this
    _KIMODO_SOURCE = "installed"
except ImportError as _e:
    # Register anyway (ComfyUI would drop the whole pack on a raise); the
    # failure then lands at KimodoLoader.load with this breadcrumb in the log.
    _KIMODO_SOURCE = None
    log.error(
        "[melite-kimodo] kimodo is NOT importable (%s) — "
        "pip install -r custom_nodes/melite-kimodo-nodes/requirements.txt",
        _e,
    )
else:
    log.info("[melite-kimodo] kimodo source: %s", _KIMODO_SOURCE)

# THE BOOT ENV ASSERT (F3 D3-06, the audio pack's sibling law): the
# kwarg-drift trap stays — transformers was unpinned ("already ships
# in the base env"), the venv is shared and last installer wins, so
# the next upstream rename would land silently and kill the motion
# lane MID-RUN again (the input_embeds burn, 2026-11-13). The tested
# bound is refused HERE, before the nodes import, in the refusal
# grammar the engine surfaces at boot.
from .env_assert import assert_transformers_bound  # noqa: E402 — before the nodes import

assert_transformers_bound()

from .nodes import (  # noqa: E402 — after the kimodo source bootstrap above
    NODE_CLASS_MAPPINGS as _KIMODO_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _KIMODO_DISPLAY,
)
from .nodes_anny import (  # noqa: E402 — the merged anny-estimator family
    NODE_CLASS_MAPPINGS as _ANNY_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _ANNY_DISPLAY,
)

NODE_CLASS_MAPPINGS = {**_KIMODO_MAPPINGS, **_ANNY_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**_KIMODO_DISPLAY, **_ANNY_DISPLAY}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
# No WEB_DIRECTORY — native widgets only; no web assets are shipped for this pack.
