"""Tech Noir Kimodo custom nodes for ComfyUI.

Wraps NVIDIA Kimodo — kinematic motion diffusion model for text-to-3D
human/humanoid motion. Replaces the now-removed kimodo pool sidecar
(Originally media/poser/app_kimodo.py + media/motion/kimodo_pose.py).

Two nodes:
  KimodoLoader    — loads the diffusion model into VRAM (~17GB)
  KimodoTextToPose — runs text-to-pose inference, saves NPZ, returns result

KIMODO SOURCE RESOLUTION (2026-08-15, "No module named 'kimodo'"):
nodes.py imports kimodo lazily (``from kimodo import load_model`` inside
KimodoLoader.load), so the pack REGISTERED fine and the failure only
surfaced mid-run, killing the whole pipeline with
"ComfyUI workflow failed: No module named 'kimodo'". The retired full
image used to pip-install the vendored source (``COPY vendor/kimodo/
/opt/kimodo/`` + ``pip install -e ".[all]"``) — that bake is GONE; the
package must come from a pip install or the owning plugin's lane
declaration. The DEV box symlinks this pack straight out of the repo and
nothing ever installed kimodo into its venv. This bootstrap closes that
gap the melite-step1x-nodes way: before registering the nodes, make the
vendored source importable — preferring an already-instolved kimodo (pip
install wins), then the repo checkout this pack was symlinked from
(realpath escapes the custom_nodes symlink to the repo's vendor/kimodo),
then the legacy /opt/kimodo layout.

requirements.txt declares the one dependency the image base env
does not already carry (hydra-core — kimodo's config instantiate()).
"""
from __future__ import annotations

import logging
import os
import sys

log = logging.getLogger("melite-kimodo")

# ── sys.path bootstrap (must happen before nodes.py's lazy kimodo imports) ──
_CANDIDATES = [
    # Dev: this pack is symlinked from <repo>/custom_nodes/melite-kimodo-nodes
    # → realpath resolves through the /opt/ComfyUI/custom_nodes symlink to
    # the REPO's live vendor tree (three levels up: ext dir → custom_nodes
    # → repo root). Ordered FIRST so dev iterates on the checkout, not a
    # stale copy.
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
        "vendor", "kimodo",
    ),
    # Legacy layout from the retired full image (it COPY'd vendor/kimodo
    # here); kept as the last-resort fallback for older installs.
    "/opt/kimodo",
]

try:
    import kimodo  # noqa: F401 — already resolvable (pip install / -e)
    _KIMODO_SOURCE = "installed"
except ImportError:
    _KIMODO_SOURCE = None
    for _p in _CANDIDATES:
        # A valid source dir carries the package itself (vendor/kimodo/kimodo).
        if os.path.isdir(os.path.join(_p, "kimodo")):
            if _p not in sys.path:
                sys.path.insert(0, _p)
            try:
                import kimodo  # noqa: F401
                _KIMODO_SOURCE = _p
                break
            except ImportError as _e:
                log.warning(
                    "[melite-kimodo] vendored source at %s failed to import (%s); "
                    "see requirements.txt — pip install -r custom_nodes/melite-kimodo-nodes/requirements.txt",
                    _p, _e,
                )
                sys.path.remove(_p)

if _KIMODO_SOURCE is None:
    # Register anyway (ComfyUI would drop the whole pack on a raise); the
    # failure then lands at KimodoLoader.load with this breadcrumb in the log.
    log.error(
        "[melite-kimodo] kimodo is NOT importable — KimodoTextToPose will fail. "
        "Expected the vendored source at one of: %s "
        "(or pip install vendor/kimodo into this venv).",
        ", ".join(_CANDIDATES),
    )
else:
    log.info("[melite-kimodo] kimodo source: %s", _KIMODO_SOURCE)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
# No WEB_DIRECTORY — native widgets only; no web assets are shipped for this pack.
