"""melite-kimodo-nodes (anny family) — image -> Anny, both engines.

Engine 1 — MultiHMR2ImageToAnny (regression, in-process):
  Wraps naver/multi-hmr2's Python API (init_hmr_session / infer_image /
  save_results_image) exactly as upstream documents it. The session is
  cached per checkpoint and registered with ComfyUI VRAM accounting via
  melite_model_base.PipelinePatcher (the DETR model AND the world body
  model are accounted; eviction moves both to CPU).

Engine 2 — AnnyFitOptimize (optimization, out-of-process):
  Wraps naver/anny-fit's two CLIs (preprocess.build_test_dataset +
  annyfit/optimize.py) as subprocesses over a side checkout, matching
  setup.sh's PYTHONPATH/PYOPENGL contract. The stage schedule is
  upstream's configs/demo/multihmr.yaml loaded and PATH-OVERRIDDEN —
  never retyped here (no copy-paste fossils); only data paths and the
  logger save_dir are rewritten, to absolute paths inside the run
  directory.

No silent defaults anywhere: a missing checkpoint, missing checkout,
missing interpreter, empty detection, or failed stage refuses LOUDLY
with the exact remediation.

Checkpoints:
  multihmr2.pt — auto-downloaded by upstream (wget, NAVER non-commercial
    license) when the resolved path does not exist yet. Resolution chain:
    node input > MULTIHMR2_CKPT env > <comfy models dir>/multihmr2/
    multihmr2.pt. The file stem MUST be "multihmr2" (upstream asserts).
  anny-fit — its own checkpoints/ inside ANNYFIT_SRC (download_checkpoints.sh
    + the INSTALL.md manual list); failures surface upstream's own error.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("melite-kimodo-nodes")

# ComfyUI core module — optional so the pack imports in standalone tooling.
try:
    import folder_paths  # type: ignore  # noqa: F401  (ComfyUI core)
except ImportError:  # standalone tooling/tests, never inside ComfyUI
    folder_paths = None  # type: ignore


# ── ComfyUI-tracked pipeline wrapper (melite_model_base) ────────────────────
try:
    from melite_model_base import (  # type: ignore
        PipelinePatcher as _PipelinePatcher,
        register_pipeline as _register_pipeline,
    )
except ImportError:
    _VENDOR_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_vendor"
    )
    if _VENDOR_DIR not in sys.path:
        sys.path.insert(0, _VENDOR_DIR)
    try:
        from melite_model_base import (  # type: ignore
            PipelinePatcher as _PipelinePatcher,
            register_pipeline as _register_pipeline,
        )
    except ImportError as _e:
        raise RuntimeError(
            "melite_model_base unavailable: neither installed nor vendored "
            "(./_vendor/melite_model_base missing or corrupt — reinstall "
            "this node pack). Original error: " + str(_e)
        ) from _e


_ANNY_TAG = "[melite-kimodo-nodes/anny]"

_MULTIHMR2_CKPT_ENV = "MULTIHMR2_CKPT"
_ANNYFIT_SRC_ENV = "ANNYFIT_SRC"
_ANNYFIT_PY_ENV = "ANNYFIT_PYTHON"
_ANNYFIT_SRC_DEFAULT = "/opt/anny-fit"


# ════════════════════════════════════════════════════════════════════════════
# The UV seat (2026-10-13, the position-only export cure): upstream's
# saver writes POSITION-only GLBs by construction (trimesh export of
# the regressed template, no TEXCOORD_0), and downstream refuses such
# a body — the coat has nowhere to land a detail map, the fit has no
# seat. The unwrap is REAL chart formation (xatlas: islands from the
# bake's own geometry), never a planar projection bolted on at the
# export; a mesh that arrives WITH UVs rides them through untouched.
# ════════════════════════════════════════════════════════════════════════════
def _unwrap_uv(mesh) -> "tuple":
    """One mesh → (vertices, faces, uv) with real island UVs.

    Carries existing TEXCOORD_0 through verbatim; otherwise unwraps
    with xatlas (chart + pack — the maintained unwrapper, a declared
    pack dependency). Loud refusal, never a silent planar fallback.

    Returns the REMESHED vertex/face arrays xatlas produced (unwrap
    may split seam vertices) together with the uv per NEW vertex.
    """
    import numpy as np

    existing = getattr(getattr(mesh, "visual", None), "uv", None)
    if existing is not None and len(existing) == len(mesh.vertices):
        return (
            np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.uint32),
            np.asarray(existing, dtype=np.float32),
        )
    try:
        import xatlas  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "xatlas is required to unwrap the anny body (the coat "
            "refuses a position-only body): pip install xatlas into "
            "the serving venv (a declared melite-kimodo-nodes "
            "requirement). Original error: " + str(e)
        ) from e
    atlas = xatlas.Atlas()
    atlas.add_mesh(
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.uint32),
    )
    atlas.generate()
    vmapping, faces, uv = atlas[0]
    verts = np.asarray(mesh.vertices, dtype=np.float32)[vmapping]
    return verts, np.asarray(faces, dtype=np.uint32), np.asarray(uv, dtype=np.float32)


# ════════════════════════════════════════════════════════════════════════════
# Shared output-directory convention (autoremesher pattern: counter under
# the ComfyUI output dir, derived from filename_prefix)
# ════════════════════════════════════════════════════════════════════════════
def _next_run_dir(filename_prefix: str) -> Path:
    if folder_paths is None:
        raise RuntimeError(
            "anny-estimator nodes require ComfyUI (folder_paths) — "
            "standalone use is not supported"
        )
    out_dir = Path(folder_paths.get_output_directory())
    base, prefix = os.path.split(filename_prefix)
    sub = base.strip("/")
    target = out_dir / sub if sub else out_dir
    target.mkdir(parents=True, exist_ok=True)
    idx = 1
    run = target / f"{prefix}_{idx:05d}"
    while run.exists():
        idx += 1
        run = target / f"{prefix}_{idx:05d}"
    run.mkdir(parents=True)
    return run


def _ui_file(path: Path) -> dict:
    """One ComfyUI ui file row, relative to the output directory."""
    out_dir = Path(folder_paths.get_output_directory())
    rel = os.path.relpath(path, out_dir)
    return {
        "filename": os.path.basename(rel),
        "subfolder": os.path.dirname(rel),
        "type": "output",
    }


# ════════════════════════════════════════════════════════════════════════════
# Engine 1: Multi-HMR 2 session (cached, VRAM-tracked)
# ════════════════════════════════════════════════════════════════════════════
class _Multihmr2Patcher(_PipelinePatcher):
    """Marks this PipelinePatcher as a Multi-HMR 2 session."""
    pipeline_type = "multihmr2-session"


class _TrackedSession:
    """Adapter so PipelinePatcher can account + move BOTH session modules.

    InferenceSession holds the DETR nn.Module (session.model) and the
    world-parameterization Anny body model (session.body_model_world),
    but has no .to() of its own. PipelinePatcher walks vars() for
    nn.Modules (both are accounted) and calls .to(device) on moves —
    this adapter implements it.
    """

    def __init__(self, session) -> None:
        self.detr = session.model
        self.body_model_world = session.body_model_world

    def to(self, device):
        self.detr.to(device)
        self.body_model_world.to(device)
        return self


_SESSION_CACHE: dict = {}


def _resolve_checkpoint(explicit: str) -> str:
    """Resolution chain: node input > MULTIHMR2_CKPT > models dir.

    The FIRST EXISTING candidate wins; if none exists, the first
    candidate is returned and upstream auto-downloads to it (wget,
    NAVER non-commercial license). Loud on a wrong stem — upstream
    asserts ckpt_path.stem == "multihmr2", so we refuse it HERE with
    a remediation instead of an upstream stack trace.
    """
    cands: list = []
    if explicit:
        cands.append(("node input", explicit))
    env = os.environ.get(_MULTIHMR2_CKPT_ENV, "").strip()
    if env:
        cands.append((_MULTIHMR2_CKPT_ENV, env))
    if folder_paths is not None:
        cands.append((
            "comfy models dir",
            os.path.join(folder_paths.models_dir, "multihmr2", "multihmr2.pt"),
        ))
    if not cands:
        raise RuntimeError(
            "MultiHMR2ImageToAnny: no checkpoint source — pass the "
            "'checkpoint' input, set MULTIHMR2_CKPT, or run inside "
            "ComfyUI (models dir candidate)"
        )
    for source, path in cands:
        if Path(path).stem != "multihmr2":
            raise RuntimeError(
                f"MultiHMR2ImageToAnny: checkpoint {path!r} (from {source}) "
                "must be named 'multihmr2.pt' — upstream asserts the stem"
            )
    for _source, path in cands:
        if os.path.isfile(path):
            return path
    source, path = cands[0]
    print(f"{_ANNY_TAG} checkpoint not found; upstream will download to "
          f"{path} (from {source})")
    return path


def _get_session(ckpt_path: str):
    """Lazy-load the Multi-HMR 2 session (cached across calls) and
    register it with ComfyUI VRAM accounting (best-effort — standalone
    use without comfy.model_management skips tracking gracefully)."""
    key = str(Path(ckpt_path).resolve())
    if key not in _SESSION_CACHE:
        import multihmr2  # noqa: PLC0415  (heavy: torch + anny)

        t0 = time.perf_counter()
        session = multihmr2.init_hmr_session(key)
        _SESSION_CACHE[key] = session
        try:
            patcher = _Multihmr2Patcher(
                _TrackedSession(session),
                name=f"multihmr2:{Path(key).parent.name}",
            )
            _register_pipeline(f"multihmr2:{key}", patcher)
            print(f"{_ANNY_TAG} session loaded from {key} "
                  f"({time.perf_counter() - t0:.1f}s, "
                  f"{patcher.model_size() / 1e6:.1f}MB tracked)")
        except Exception as e:  # tracking must never block inference
            logger.warning("VRAM tracking unavailable for multihmr2 "
                           "session: %s", e)
    return _SESSION_CACHE[key]


class MultiHMR2ImageToAnny:
    """Multi-HMR 2 regression: image -> multi-person Anny params + meshes.

    In-process, VRAM-tracked. Outputs land under one run directory in
    the ComfyUI output tree:
      anny_params/<stem>.pkl      Anny params, world parameterization
                                  (K, shape, pose_parameters — upstream's
                                  --save_anny_params format)
      meshes/<stem>_NNN.glb       one GLB per detected person
      visu/<stem>.png             optional overlay render (render=True)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conf_thresh": ("FLOAT", {
                    "default": 0.4, "min": 0.05, "max": 0.95, "step": 0.01,
                    "tooltip": "Minimum confidence to keep a person",
                }),
                "dist_thresh_nms": ("FLOAT", {
                    "default": 0.25, "min": 0.05, "max": 1.0, "step": 0.01,
                    "tooltip": "Pelvis-distance threshold (m) for 3D NMS",
                }),
                "lowres": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Low-resolution Anny body (613 verts)",
                }),
                "save_anny_params": ("BOOLEAN", {"default": True}),
                "save_mesh": ("BOOLEAN", {"default": True}),
                "render": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Overlay render (needs the multihmr2 render "
                               "extras: pyrender)",
                }),
                "filename_prefix": ("STRING", {"default": "anny/multihmr2"}),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Image tensor (H,W,C in 0..1) — batch index 0",
                }),
                "image_path": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Path to an image file (used when no tensor "
                               "is wired)",
                }),
                "checkpoint": ("STRING", {
                    "default": "", "multiline": False, "forceInput": True,
                    "tooltip": "multihmr2.pt path — defaults to "
                               "MULTIHMR2_CKPT env > models dir",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("out_dir", "anny_params_path", "mesh_paths")
    FUNCTION = "estimate"
    CATEGORY = "TechNoir/Anny"
    OUTPUT_NODE = True

    def estimate(
        self,
        conf_thresh: float,
        dist_thresh_nms: float,
        lowres: bool,
        save_anny_params: bool,
        save_mesh: bool,
        render: bool,
        filename_prefix: str,
        image=None,
        image_path: str = "",
        checkpoint: str = "",
    ):
        import numpy as np  # noqa: PLC0415

        # ── resolve the image source: tensor > path; loud if neither ──
        stem = "image"
        if image is not None:
            frame = image[0].clamp(0, 1).cpu().numpy()
            img_array = (frame * 255.0).round().astype(np.uint8)  # H,W,C
        elif image_path.strip():
            p = Path(image_path.strip())
            if not p.is_file():
                raise RuntimeError(
                    f"MultiHMR2ImageToAnny: image not found ({p})"
                )
            stem = p.stem
            img_array = None
        else:
            raise RuntimeError(
                "MultiHMR2ImageToAnny: no input — wire 'image' or set "
                "'image_path'"
            )

        if not (save_anny_params or save_mesh or render):
            raise RuntimeError(
                "MultiHMR2ImageToAnny: every output is disabled — enable "
                "save_anny_params, save_mesh or render"
            )

        run_dir = _next_run_dir(filename_prefix)
        session = _get_session(_resolve_checkpoint(checkpoint))
        import multihmr2  # noqa: PLC0415

        t0 = time.perf_counter()
        if img_array is not None:
            session.current_img_name = stem  # np branch never sets it
            pred = session(
                img_array,
                conf_thresh=conf_thresh,
                dist_thresh_nms=dist_thresh_nms,
                lowres=lowres,
            )
            render_src = None
            if render:
                from PIL import Image  # noqa: PLC0415

                staged = run_dir / f"{stem}_input.png"
                Image.fromarray(img_array).save(staged)
                render_src = str(staged)
        else:
            pred = multihmr2.infer_image(
                session,
                str(Path(image_path.strip())),
                conf_thresh=conf_thresh,
                dist_thresh_nms=dist_thresh_nms,
                lowres=lowres,
            )
            render_src = image_path.strip() if render else None

        n_persons = int(getattr(pred.persons, "num_person", len(pred)))
        print(f"{_ANNY_TAG} {stem}: {n_persons} person(s) "
              f"({time.perf_counter() - t0:.1f}s)")
        if n_persons == 0:
            raise RuntimeError(
                f"MultiHMR2ImageToAnny: no person detected in {stem!r} at "
                f"conf_thresh={conf_thresh} — lower the threshold or check "
                "the image"
            )

        multihmr2.save_results_image(
            session,
            pred,
            str(run_dir),
            save_mesh=save_mesh,
            save_anny_params=save_anny_params,
            lowres=lowres,
        )
        if render:
            if not render_src:
                raise RuntimeError("render requested without an image source")
            multihmr2.render_results_image(
                session, pred, render_src, str(run_dir), lowres=lowres
            )

        params_path = run_dir / "anny_params" / f"{session.current_img_name}.pkl"
        meshes = sorted((run_dir / "meshes").glob("*.glb")) if save_mesh else []
        visu = sorted((run_dir / "visu").glob("*.png")) if render else []

        if save_anny_params and not params_path.is_file():
            raise RuntimeError(
                f"MultiHMR2ImageToAnny: expected {params_path} was not "
                "written — upstream save_results_image failed silently"
            )
        if save_mesh and not meshes:
            raise RuntimeError(
                "MultiHMR2ImageToAnny: no mesh GLBs were written — "
                "is trimesh installed in the serving venv?"
            )

        ui: dict = {
            "three_model": [_ui_file(m) for m in meshes],
            "images": [_ui_file(v) for v in visu],
        }
        return {
            "ui": ui,
            "result": (
                str(run_dir),
                str(params_path) if save_anny_params else "",
                ";".join(str(m) for m in meshes),
            ),
        }


# ════════════════════════════════════════════════════════════════════════════
# Graph-side single-person selection (the card wiring seam)
# ════════════════════════════════════════════════════════════════════════════
class AnnySelectPerson:
    """Pick ONE person's GLB out of a multi-person estimator run.

    MultiHMR2ImageToAnny returns mesh_paths as a ';'-joined STRING (one
    GLB per detected person, detection order). Downstream character
    assembly (CompositeCharacter) takes exactly ONE path — this node is
    the loud bridge: it refuses empty estimator output and out-of-range
    indices with the full detected list, never silently truncating.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_paths": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "';'-joined GLB paths — wire "
                               "MultiHMR2ImageToAnny.mesh_paths",
                }),
                "index": ("INT", {
                    "default": 0, "min": 0, "max": 64,
                    "tooltip": "Which detected person (detection "
                               "order, 0 = first)",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "select"
    CATEGORY = "TechNoir/Anny"

    @staticmethod
    def select(mesh_paths: str, index: int):
        paths = [p.strip() for p in (mesh_paths or "").split(";")
                 if p.strip()]
        if not paths:
            raise RuntimeError(
                "AnnySelectPerson: estimator returned no mesh paths — "
                "the run detected nobody (or produced no GLBs). Feed a "
                "photo with a visible person."
            )
        if index >= len(paths):
            raise RuntimeError(
                f"AnnySelectPerson: index {index} out of range — "
                f"{len(paths)} person(s) detected: "
                + "; ".join(paths)
            )
        return (paths[index],)


class AnnyGroundBody:
    """Ground an estimator body mesh for card assembly.

    MultiHMR2ImageToAnny's meshes are OpenCV CAMERA-SPACE (+X right,
    +Y DOWN, +Z forward; the person floats at camera distance —
    live-probed 2026-09-05: extent ~[0.3, 0.45, 0.4] centered z≈1.85,
    body-up = −Y). Character assembly (CompositeCharacter → turnaround
    renders) wants a WORLD body: +Y up, feet on the ground plane,
    origin-centered. This node applies the KNOWN camera→world map
    (rotate π about Z: x→−x, y→−y — right-handed, flips the down-axis
    to up), grounds the feet, centers X/Z, and writes to output/.

    Orientation is NOT guessed from extents (a y-tallest camera-space
    body is upside-down — extent can't see that); the pack knows its
  estimator's convention. A post-flip guard asserts the result is
    plausibly a standing figure (y-extent is the tallest, within 2×)
    and refuses loudly otherwise. Scale is preserved (metres in,
    metres out).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_path": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "estimator mesh GLB path — wire "
                               "AnnySelectPerson.glb_path",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "ground"
    OUTPUT_NODE = True          # terminal artifact: must execute
    CATEGORY = "TechNoir/Anny"

    @staticmethod
    def ground(glb_path: str):
        import trimesh

        p = (glb_path or "").strip()
        if not p or not os.path.isfile(p):
            raise RuntimeError(
                f"AnnyGroundBody: '{glb_path}' is not a file — wire "
                "AnnySelectPerson.glb_path (an estimator mesh GLB)."
            )
        scene = trimesh.load(p, force="scene")
        geoms = [g for g in scene.geometry.values()
                 if hasattr(g, "vertices") and len(g.vertices)]
        if not geoms:
            raise RuntimeError(
                f"AnnyGroundBody: '{p}' carries no mesh geometry."
            )
        import numpy as np
        verts = np.vstack([np.asarray(g.vertices, dtype=np.float64)
                           for g in geoms])
        faces, offset = [], 0
        for g in geoms:
            f = np.asarray(g.faces, dtype=np.int64)
            faces.append(f + offset)
            offset += len(g.vertices)
        # THE UV CARRY (the position-only cure): merge each geom's
        # TEXCOORD_0 alongside the vertices when EVERY geom ships one
        # (upstream's saver ships none — the unwrap below answers).
        uvs = [getattr(getattr(g, "visual", None), "uv", None) for g in geoms]
        merged_uv = None
        if all(u is not None and len(u) == len(g.vertices)
               for u, g in zip(uvs, geoms)):
            merged_uv = np.vstack([np.asarray(u, dtype=np.float64)
                                   for u in uvs])
        visual = (None if merged_uv is None
                  else trimesh.visual.TextureVisuals(uv=merged_uv))
        mesh = trimesh.Trimesh(
            vertices=verts, faces=np.vstack(faces), process=False,
            visual=visual)

        # OpenCV camera space -> world: y-down becomes y-up (π about Z,
        # right-handed), then drop the camera translation (re-anchor).
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1]))
        v = np.asarray(mesh.vertices, dtype=np.float64)
        v[:, 1] -= v[:, 1].min()           # feet on the ground
        v[:, 0] -= (v[:, 0].max() + v[:, 0].min()) / 2
        v[:, 2] -= (v[:, 2].max() + v[:, 2].min()) / 2

        extent = v.max(axis=0) - v.min(axis=0)
        if extent[1] < 0.5 * max(extent[0], extent[2]):
            raise RuntimeError(
                "AnnyGroundBody: grounded mesh is not plausibly a "
                f"standing figure (extent {extent.round(3).tolist()}) — "
                "up axis is not tallest after the camera→world map."
            )
        out_dir = Path(folder_paths.get_output_directory()) / "anny"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / ("anny_grounded_body_%06d.glb" % int(time.time()))
        # THE UNWRAP SEAT: real island UVs on the FINAL grounded body
        # (xatlas charts, or the merged TEXCOORD_0 carried verbatim) —
        # the export never writes a position-only body again.
        grounded = trimesh.Trimesh(vertices=v, faces=mesh.faces,
                                   process=False,
                                   visual=mesh.visual)
        uv_verts, uv_faces, uv = _unwrap_uv(grounded)
        trimesh.Trimesh(
            vertices=uv_verts, faces=uv_faces, process=False,
            visual=trimesh.visual.TextureVisuals(uv=uv),
        ).export(str(out))
        logger.info("[AnnyGroundBody] %s -> %s (extent %s)",
                    p, out, extent.round(3))
        return {
            "result": (str(out),),
            "ui": {"three_model": [_ui_file(out)]},
        }


# ════════════════════════════════════════════════════════════════════════════
# Engine 2: Anny-Fit optimization (out-of-process subprocess)
# ════════════════════════════════════════════════════════════════════════════
def _annyfit_root() -> Path:
    src = Path(os.environ.get(_ANNYFIT_SRC_ENV, _ANNYFIT_SRC_DEFAULT))
    expected = [src / "preprocess" / "build_test_dataset.py",
                src / "annyfit" / "optimize.py"]
    missing = [str(p) for p in expected if not p.is_file()]
    if missing:
        raise RuntimeError(
            "anny-fit checkout not found at " + str(src) + ". Install it:\n"
            "  git clone --recurse-submodules "
            "https://github.com/naver/anny-fit " + str(src) + "\n"
            "  cd " + str(src) + " && bash scripts/install.sh && "
            "bash scripts/download_checkpoints.sh\n"
            "  + the INSTALL.md manual checkpoints "
            "(ViTPose/CameraHMR/SMPL)\n"
            "Missing: " + ", ".join(missing) + ".\n"
            "or point " + _ANNYFIT_SRC_ENV + " at an existing checkout."
        )
    return src


def _annyfit_python(src: Path) -> str:
    py = os.environ.get(_ANNYFIT_PY_ENV, "").strip() or str(
        src / ".venv" / "bin" / "python"
    )
    if not os.path.isfile(py):
        raise RuntimeError(
            f"anny-fit interpreter not found at {py}. Create the env "
            f"(scripts/install.sh inside the checkout) or set {_ANNYFIT_PY_ENV} "
            "to the venv python that has anny-fit's requirements installed."
        )
    return py


def _annyfit_env(src: Path) -> dict:
    """setup.sh's contract: PYTHONPATH over the checkout + submodules,
    EGL rendering (osmesa override honored from the outside)."""
    env = dict(os.environ)
    py_path = os.pathsep.join(str(p) for p in (
        src,
        src / "submodules" / "multi-hmr",
        src / "submodules" / "CameraHMR",
    ))
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = py_path + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = py_path
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    return env


def _stage_images(image_path: str, work: Path) -> int:
    """Copy input images into work/images/ (anny-fit's data_root layout).

    Accepts a single image file, a folder of images, or a data_root-style
    folder containing images/. Staging (never mutating the source) keeps
    anny-fit's preprocessing writes inside the run directory.
    """
    p = Path(image_path.strip())
    if not p.exists():
        raise RuntimeError(f"AnnyFitOptimize: input not found ({p})")
    img_dir = work / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    if p.is_file():
        files = [p]
    else:
        sub = p / "images"
        root = sub if sub.is_dir() else p
        files = sorted(
            f for f in root.iterdir()
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
    if not files:
        raise RuntimeError(
            f"AnnyFitOptimize: no .jpg/.jpeg/.png found under {p}"
        )
    for f in files:
        shutil.copy2(f, img_dir / f.name)
    return len(files)


def _run_stage(cmd: list, cwd: Path, env: dict, timeout: int, what: str) -> None:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), env=env, capture_output=True,
            text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"anny-fit {what} timed out after {timeout}s — raise the "
            f"timeout knob ({e})"
        ) from e
    if proc.returncode != 0:
        raise RuntimeError(
            f"anny-fit {what} failed (exit {proc.returncode}):\n"
            f"--- stderr tail ---\n{proc.stderr[-3000:]}\n"
            f"--- stdout tail ---\n{proc.stdout[-1500:]}"
        )
    print(f"{_ANNY_TAG} {what} ok ({time.perf_counter() - t0:.1f}s)")


def _derive_config(src: Path, work: Path, dataset: str, vlm_version: str,
                   multihmr_model: str) -> Path:
    """Load upstream's demo config and OVERRIDE PATHS ONLY.

    The stage schedule / loss weights stay upstream's SSOT — retyping
    them here would be a copy-paste fossil that drifts. Only
    data.dataset_folder, data.mesh_folder and logger.save_dir are
    rewritten, to absolute paths inside the run directory.
    """
    import yaml  # noqa: PLC0415

    base = src / "annyfit" / "configs" / "demo" / "multihmr.yaml"
    cfg = yaml.safe_load(base.read_text()) or {}
    # upstream: num_people = 10 for multi_person, 1 for single_person
    max_people = 10 if dataset == "multi_person" else 1
    cfg.setdefault("data", {})["dataset_folder"] = str(
        work / f"person_dataset_{max_people}_{vlm_version}"
    )
    cfg["data"]["mesh_folder"] = str(work / f"multihmr_{multihmr_model}")
    cfg.setdefault("logger", {})["save_dir"] = str(work / "results")
    cfg["logger"]["name"] = "annyfit"
    out = work / "annyfit_config.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


class AnnyFitOptimize:
    """Anny-Fit camera-space optimization: images -> refined Anny params.

    Runs the full anny-fit pipeline out-of-process over the side
    checkout (its deps never touch the ComfyUI venv):
      1. preprocess.build_test_dataset (detect + ViTPose + camera +
         dense keypoints + VLM attributes + MultiHMR-Anny init)
      2. annyfit/optimize.py with a path-overridden demo config
    Outputs land in the run directory:
      results/annyfit/version_<stem>/params.npz      refined per-person
                                                    Anny params
      results/annyfit/version_<stem>/vis/*.jpg       initial/final renders
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Image file, folder of images, or a "
                               "data_root folder containing images/",
                }),
                "dataset": (["multi_person", "single_person"],
                            {"default": "multi_person"}),
                "detector": (["groundingdino", "detectron2"],
                             {"default": "groundingdino"}),
                "vlm_version": (["headcrop", "crop", "maskedcrop"],
                                {"default": "headcrop"}),
                "preprocess_timeout_s": ("INT", {
                    "default": 3600, "min": 60, "max": 86400,
                    "tooltip": "Per-stage subprocess timeout (preprocess)",
                }),
                "optimize_timeout_s": ("INT", {
                    "default": 7200, "min": 60, "max": 86400,
                    "tooltip": "Per-stage subprocess timeout (optimize)",
                }),
                "filename_prefix": ("STRING", {"default": "anny/annyfit"}),
            },
            "optional": {
                "multihmr_model": ("STRING", {
                    "default": "multiHMR_672_L_anny",
                    "tooltip": "Init checkpoint name inside the checkout's "
                               "checkpoints/ dir",
                }),
                "skip_camera": ("BOOLEAN", {
                    "default": False, "tooltip": "--no_camera",
                }),
                "skip_densekp": ("BOOLEAN", {
                    "default": False, "tooltip": "--no_densekp",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("out_dir", "anny_params_paths", "render_paths")
    FUNCTION = "optimize"
    CATEGORY = "TechNoir/Anny"
    OUTPUT_NODE = True

    def optimize(
        self,
        image_path: str,
        dataset: str,
        detector: str,
        vlm_version: str,
        preprocess_timeout_s: int,
        optimize_timeout_s: int,
        filename_prefix: str,
        multihmr_model: str = "multiHMR_672_L_anny",
        skip_camera: bool = False,
        skip_densekp: bool = False,
    ):
        if not image_path.strip():
            raise RuntimeError(
                "AnnyFitOptimize: image_path is required — a file, a folder "
                "of images, or a data_root with images/"
            )
        src = _annyfit_root()
        py = _annyfit_python(src)
        env = _annyfit_env(src)
        run_dir = _next_run_dir(filename_prefix)
        n_imgs = _stage_images(image_path, run_dir)
        print(f"{_ANNY_TAG} anny-fit {src} | {n_imgs} image(s) staged "
              f"-> {run_dir}")

        # ── stage 1: preprocessing (detection -> dataset npz files) ──
        cmd = [
            py, "-m", "preprocess.build_test_dataset",
            "--data_root", str(run_dir),
            "--dataset_name", dataset,
            "--preprocess_data",
            "--detector", detector,
            "--vlm_version", vlm_version,
            "--multihmr_model", multihmr_model,
        ]
        if skip_camera:
            cmd.append("--no_camera")
        if skip_densekp:
            cmd.append("--no_densekp")
        _run_stage(cmd, src, env, preprocess_timeout_s, "preprocess")

        # ── stage 2: staged optimization (path-overridden demo config) ──
        cfg_path = _derive_config(
            src, run_dir, dataset, vlm_version, multihmr_model
        )
        _run_stage(
            [py, "optimize.py", "--config", str(cfg_path)],
            src / "annyfit", env, optimize_timeout_s, "optimize",
        )

        # ── collect: per-image version dirs under results/annyfit ──
        res_root = run_dir / "results" / "annyfit"
        params = sorted(res_root.glob("version_*/params.npz"))
        renders = sorted(res_root.glob("version_*/vis/*.jpg"))
        if not params:
            tree = "\n".join(str(p) for p in sorted(run_dir.rglob("*"))[:40])
            raise RuntimeError(
                f"anny-fit optimize produced no params.npz under "
                f"{res_root}. Run tree:\n{tree}"
            )

        ui = {"images": [_ui_file(r) for r in renders]}
        return {
            "ui": ui,
            "result": (
                str(run_dir),
                ";".join(str(p) for p in params),
                ";".join(str(r) for r in renders),
            ),
        }


# ════════════════════════════════════════════════════════════════════════════
# Registration
# ════════════════════════════════════════════════════════════════════════════
_COMMERCIAL_ROOT_ENV = "ANNY_COMMERCIAL_ROOT"
_COMMERCIAL_ROOT_REL = ("data", "runtime", "anny-commercial")
# THE ROOT SEARCH. The managed boot's convention put ComfyUI's output
# dir at <ws>/data/temp, so parents[1] was the workspace — a BYO
# deployment (ComfyUI living at <repo>/data/runtime/comfyui) lands
# that one level too deep and the join doubles into
# data/runtime/data/runtime. So: walk UP from the output directory and
# take the first ancestor that actually holds the runtime. The env
# var still wins (the refusal below names it), and a machine with no
# runtime anywhere falls through to the managed-boot guess so the
# refusal quotes the conventional path.
def _default_commercial_root() -> str:
    if not folder_paths:
        return ""
    try:
        start = Path(folder_paths.get_output_directory()).resolve()
    except Exception:
        return ""
    managed = start.parents[1] / "data" / "runtime" / "anny-commercial"
    for parent in start.parents:
        candidate = parent.joinpath(*_COMMERCIAL_ROOT_REL)
        if candidate.is_dir():
            return str(candidate)
    return str(managed)


_COMMERCIAL_ROOT_DEFAULT = _default_commercial_root()


def _commercial_root() -> Path:
    """The provisioned commercial-chain runtime (venv + SOMA-X + worker).

    data/runtime/anny-commercial/ is materialized by the estate
    provisioner (torch cu130 — the system nvcc 13.x cannot build
    against the serving venv's cu128 stack, so the chain rides its
    own venv as a subprocess, the AnnyFitOptimize pattern).
    """
    src = Path(os.environ.get(_COMMERCIAL_ROOT_ENV, _COMMERCIAL_ROOT_DEFAULT))
    expected = [src / "commercial_chain.py",
                src / ".venv" / "bin" / "python",
                src / "SOMA-X" / "tools" / "mhr2soma.py"]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise RuntimeError(
            "anny-commercial runtime not found at " + str(src) + ". "
            "Provision it: uv run python tools/provision.py commercial "
            "(SAM 3D Body venv + SOMA-X checkout — see "
            "docs/design/commercial-estimator-chain.md). Missing: "
            + ", ".join(missing) + ". Or point " + _COMMERCIAL_ROOT_ENV
            + " at an existing runtime."
        )
    return src


class Sam3dBodyImageToAnny:
    """COMMERCIAL chain: photo → SAM 3D Body → MHR → SOMA → Anny → GLB.

    Licensing (design record docs/design/commercial-estimator-chain.md,
    operator order 2026-09-05): SAM 3D Body rides the Meta SAM License
    — ACCEPTED for commercial games use ("I CAN SELL IT, I just have
    to remove it if they ask"); MHR + SOMA-X + anny are Apache 2.0.
    This lane NEVER imports multihmr2/AnnyFit (NAVER non-commercial —
    those stay in the research lane).

    Out-of-process: the heavy stack (torch cu130 + detectron2) rides
    the provisioned anny-commercial venv, spawned per run with
    PROGRESS streaming; every stage failure surfaces its full stderr.
    Artifacts in the run dir: commercial_body.glb (grounded Y-up
    bind-pose body fitted to the photo's person), anny_params.npz
    (phenotype + local-change rows — card/slider vocabulary),
    soma.npz, sam3d.parquet.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Image tensor (H,W,C in 0..1) — batch "
                               "index 0",
                }),
            },
            "optional": {
                "image_path": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Direct photo path (overrides the tensor; "
                               "no recompression)",
                }),
                "person_index": ("INT", {
                    "default": 0, "min": 0, "max": 32,
                    "tooltip": "Which detected person (detection order)",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("glb_path", "anny_params_path")
    FUNCTION = "estimate"
    OUTPUT_NODE = True
    CATEGORY = "TechNoir/Anny"

    @staticmethod
    def estimate(image, image_path="", person_index=0):
        root = _commercial_root()
        run_dir = _next_run_dir("commercial_anny")

        src = (image_path or "").strip()
        if src:
            if not os.path.isfile(src):
                raise RuntimeError(
                    f"Sam3dBodyImageToAnny: image_path '{src}' is not a "
                    "file.")
        else:
            if image is None:
                raise RuntimeError(
                    "Sam3dBodyImageToAnny: no input — wire 'image' or set "
                    "'image_path'.")
            import numpy as npv
            from PIL import Image as PILImage
            arr = (npv.asarray(image[0].cpu()) * 255.0
                   ).clip(0, 255).astype("uint8")
            src = run_dir / "input.png"
            PILImage.fromarray(arr).save(str(src))

        proc = subprocess.run(
            [str(root / ".venv" / "bin" / "python"),
             str(root / "commercial_chain.py"), str(src), str(run_dir)],
            capture_output=True, text=True,
            # the cu130 nvidia libs are NOT on the loader path by
            # default — torch's nvrtc JIT (warp kernels in mhr2soma)
            # fails to open libnvrtc-builtins.so.13.0 without this
            # (live-probed 2026-09-05).
            env={**os.environ,
                 "LD_LIBRARY_PATH": str(
                     root / ".venv" / "lib" / "python3.11"
                     / "site-packages" / "nvidia" / "cu13" / "lib"),
                 "PERSON_INDEX": str(person_index)},
            cwd=str(root))
        for line in proc.stdout.splitlines():
            if line.startswith("PROGRESS"):
                logger.info("[commercial] %s", line)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Sam3dBodyImageToAnny: commercial chain failed (rc="
                f"{proc.returncode}):\n{proc.stderr[-2000:]}")
        glb = run_dir / "commercial_body.glb"
        npz = run_dir / "anny_params.npz"
        if not glb.is_file() or not npz.is_file():
            raise RuntimeError(
                "Sam3dBodyImageToAnny: chain exited 0 but artifacts are "
                f"missing ({glb}, {npz}) — worker drifted; run dir "
                f"{run_dir} retains intermediates for audit.")
        # Recipe harvest (2026-09-06): the chain's anny_preset.json
        # (estimated dialect — phenotype[6] + scale[139], written by
        # commercial_chain step 4b) rides the ui surface beside the
        # GLB so the gateway harvest downloads it as a run artifact.
        # The studio seeds its sculpt scratchpad from it (the blue
        # figure wakes up AS the estimated body); the bake path is
        # untouched (GLB still first-class, npz still required).
        preset = run_dir / "anny_preset.json"
        ui_files = [_ui_file(glb)]
        if preset.is_file():
            ui_files.append(_ui_file(preset))
        return {
            "result": (str(glb), str(npz)),
            "ui": {"three_model": ui_files},
        }


class AnnyParamsToGlb:
    """Preset params → grounded Anny body GLB (stroke 3: the preset-load
    verb's flow runs this out-of-process in the provisioned
    anny-commercial runtime — the Sam3dBodyImageToAnny pattern: the
    serving venv has anny but NOT soma/SOMA-X, and the estimated
    dialect re-instantiates through the SAME AnnySimplified path the
    commercial worker bakes).

    params_json is the canonical node dialect (CharacterPreset.
    canonical_json): authored {11 phenotype + named morphs} or
    estimated {6 + 139 verbatim estimator output}. Shape-checked
    HERE (counts + dialect — the estate row validated at
    construction; a hand-posted body gets the same gate, loud).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "params_json": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Canonical preset params (the presets "
                               "door row verbatim — never hand-edited).",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "bake"
    OUTPUT_NODE = True          # terminal artifact: must execute
    CATEGORY = "TechNoir/Anny"

    @staticmethod
    def bake(params_json: str):
        import json as _json
        try:
            doc = _json.loads(params_json or "")
        except ValueError:
            raise RuntimeError(
                "AnnyParamsToGlb: params_json is not JSON — post the "
                "presets-door row verbatim.")
        if not isinstance(doc, dict):
            raise RuntimeError("AnnyParamsToGlb: params root not an object")
        dialect = doc.get("dialect")
        ph = doc.get("phenotype")
        if dialect == "authored":
            if not isinstance(ph, list) or len(ph) != 11 \
                    or not isinstance(doc.get("local_changes"), dict):
                raise RuntimeError(
                    "AnnyParamsToGlb: authored wants phenotype[11] + "
                    "local_changes object")
        elif dialect == "estimated":
            if not isinstance(ph, list) or len(ph) != 6 \
                    or not isinstance(doc.get("scale"), list) \
                    or len(doc["scale"]) != 139:
                raise RuntimeError(
                    "AnnyParamsToGlb: estimated wants phenotype[6] + "
                    "scale[139]")
        else:
            raise RuntimeError(
                "AnnyParamsToGlb: unknown dialect %r (want "
                "authored|estimated)" % (dialect,))
        root = _commercial_root()
        script = root / "bake_params.py"
        if not script.is_file():
            raise RuntimeError(
                "AnnyParamsToGlb: bake_params.py missing in the runtime "
                "(re-provision: uv run python tools/provision.py "
                "commercial).")
        run_dir = _next_run_dir("preset_bake")
        params_path = run_dir / "params.json"
        params_path.write_text(params_json, encoding="utf-8")
        out = run_dir / "preset_body.glb"
        proc = subprocess.run(
            [str(root / ".venv" / "bin" / "python"),
             str(script), str(params_path), str(out)],
            capture_output=True, text=True,
            env={**os.environ,
                 "LD_LIBRARY_PATH": str(
                     root / ".venv" / "lib" / "python3.11"
                     / "site-packages" / "nvidia" / "cu13" / "lib")},
            cwd=str(root))
        for line in proc.stdout.splitlines():
            if line.startswith("PROGRESS"):
                logger.info("[preset-bake] %s", line)
        if proc.returncode != 0:
            raise RuntimeError(
                "AnnyParamsToGlb: bake failed (rc="
                f"{proc.returncode}):\n{proc.stderr[-2000:]}")
        if not out.is_file():
            raise RuntimeError(
                "AnnyParamsToGlb: bake exited 0 but no GLB — run dir "
                f"{run_dir} retains intermediates for audit.")
        return {
            "result": (str(out),),
            "ui": {"three_model": [_ui_file(out)]},
        }


NODE_CLASS_MAPPINGS = {
    "MultiHMR2ImageToAnny": MultiHMR2ImageToAnny,
    "AnnySelectPerson": AnnySelectPerson,
    "AnnyGroundBody": AnnyGroundBody,
    "AnnyFitOptimize": AnnyFitOptimize,
    "Sam3dBodyImageToAnny": Sam3dBodyImageToAnny,
    "AnnyParamsToGlb": AnnyParamsToGlb,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiHMR2ImageToAnny": "🧍 Multi-HMR 2 · Image → Anny",
    "AnnySelectPerson": "🧍→1 Select Person · estimator → one GLB",
    "AnnyGroundBody": "🧍↓ Ground Body · camera-space → world",
    "AnnyFitOptimize": "🎯 Anny-Fit · camera-space refine",
    "Sam3dBodyImageToAnny": "🧍💰 SAM 3D Body · commercial chain → Anny",
    "AnnyParamsToGlb": "🧍📋 Params → Body · preset instantiate",
}
