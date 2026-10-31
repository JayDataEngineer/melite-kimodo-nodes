"""Kimodo text-to-3D-motion — ComfyUI-native custom nodes.

Compliant with the 4 ComfyUI memory rules (see AGENTS.md):
  Rule 1: Loader wraps raw nn.Module in comfy.model_patcher.ModelPatcher.
          Model is instantiated on the offload device (CPU); ComfyUI owns
          the load_device (CUDA) transition.
  Rule 2: Inference calls comfy.model_management.load_models_gpu([patcher])
          BEFORE any GPU work. This moves the entire model — parameters
          AND non-persistent buffers (betas_base, alphas_cumprod_base) —
          to load_device atomically. No device mismatch is possible.
  Rule 3: Clone before modifying. We do not apply patches, so we access
          the underlying module directly via patcher.model.
  Rule 4: patcher.load_device is the source of truth for tensor placement.

This replaces the prior architecture that violated Rules 1+2:
  - media/motion/kimodo_pose.py loaded the model with device="cuda:0"
    (manual .to(cuda) — dark matter ComfyUI cannot track).
  - A custom _PipelinePatcher subclass (melite_model_base.PipelinePatcher)
    was used instead of comfy.model_patcher.ModelPatcher, so ComfyUI's
    normal load_models_gpu machinery never fired.
  - KimodoTextToPose called svc.generate() with no load_models_gpu, so
    when LRU eviction detached parts of the model between runs, the
    diffusion buffers (betas_base, persistent=False) drifted to CPU
    while the U-Net went back to CUDA, causing RuntimeError in
    space_timesteps ("Expected all tensors to be on the same device").

The ComfyUI path also lets us delete the bespoke service singleton,
the network-patch shim (now inlined verbatim), and the manual VRAM
plumbing.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("melite-kimodo")


class _InterruptibleBar:
    """Wraps kimodo's per-step loop with ComfyUI's interrupt check.

    kimodo's sampling loop (``for i in progress_bar(indices)`` in
    kimodo_model._generate) never looks at ComfyUI's processing-interrupted
    flag, so a plain /interrupt would only land when generation completes —
    the Motion Card's Freeze button would appear dead. This bar checks the
    flag BEFORE EVERY STEP and raises ComfyUI's InterruptProcessingException,
    killing the prompt promptly. The runner's stop callback surfaces the
    interrupt as a stopped run with Resume (prior artifact preserved).

    If the comfy import fails, the bar degrades to plain iteration (freeze
    falls back to a brick-end abort) — never a crash.
    """

    def __init__(self):
        self._it = None
        self._check = None
        try:
            import comfy.model_management as _mm
            self._check = _mm.throw_exception_if_processing_interrupted
        except Exception as e:  # pragma: no cover — import-path fallback
            log.warning("_InterruptibleBar: comfy interrupt check unavailable (%s)", e)

    # tqdm surface — the vendor model only ever does `for i in bar(indices)`.
    def __call__(self, iterable):
        self._it = iter(iterable)
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if self._check is not None:
            self._check()
        return next(self._it)

    def update(self, _n=1):  # pragma: no cover — defensive tqdm API
        pass

    def close(self):  # pragma: no cover — defensive tqdm API
        pass


# THE ESTATE MODEL SEAT (2026-11-01, §3.14 conformance): there is NO
# node-side model default. The default lives in the estate DECL
# (packages/models/kimodo lane_inputs model_path) and the composer
# passes it as model_name on every graph — an empty model_name refuses
# loud at the load door (see KimodoLoader.load). KIMODO_MODEL is not
# read here: locations never come from process env in this pack.
DEFAULT_FPS = 30
DEFAULT_DURATION_SEC = 6.0
DEFAULT_DIFFUSION_STEPS = 100
DEFAULT_NUM_SAMPLES = 1
DEFAULT_NUM_TRANSITION_FRAMES = 30
DEFAULT_CFG_WEIGHT = [2.0, 2.0]
DEFAULT_CFG_TYPE = "separated"


def _b64_to_f32(b64: str) -> np.ndarray:
    """Decode a base64 little-endian float32 string → numpy array.

    Round-trips web/editor/.../constraints.ts encodeF32 (which writes
    ``Float32Array`` bytes via encodeBufferToB64). dtype='<f4' is the numpy
    spelling of the TS Float32Array's native byte order. NO SILENT FALLBACK:
    a corrupt payload raises (the user authored real constraints; dropping
    them silently would make the bar appear to work while doing nothing)."""
    return np.frombuffer(base64.b64decode(b64), dtype="<f4")


def _build_user_constraints(
    motion_constraints: str,
    model_skel,
    device,
) -> list:
    """Deserialize the editor's timeline constraints → kimodo constraint objs.

    Mirrors vendor/kimodo/kimodo/demo/generation.py:24-111
    (compute_model_constraints_lst) — the demo builds these objects in-process
    from session.constraints; we deserialize the same data shape from JSON
    because the editor is remote from the kimodo library. The byte contract
    is identical: jp/jr are base64 '<f4', SOMA-77 joint order, meters.

    Returns a flat list of FullBodyConstraintSet / EndEffectorConstraintSet /
    Root2DConstraintSet objects (one per fullbody track + one per EE joint
    that has keyframes + one for root2d). The caller replicates the list per
    sample into constraint_lst. Returns an empty list if no track carried
    keyframes — the caller then leaves constraint_lst untouched (no-op, not a
    crash)."""
    from kimodo.constraints import (
        EndEffectorConstraintSet,
        FullBodyConstraintSet,
        Root2DConstraintSet,
    )
    from kimodo.skeleton.definitions import SOMASkeleton77

    try:
        data = json.loads(motion_constraints)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"KimodoTextToPose: motion_constraints JSON parse failed: {exc}. "
            f"The editor serialized this — a corrupt payload is a real bug, "
            f"not a recoverable state (NO SILENT FALLBACK)."
        ) from exc

    # Skeleton remap: the editor snapshots SOMA-77 poses (77 joints). The
    # model denoises on its internal skeleton (typically SOMASkeleton30).
    # Same slice as the chain path (nodes.py:434-439) and the demo bridge
    # (generation.py:38,73-75). Guard with nbjoints so a future 30-joint
    # model passes through untouched.
    skel_slice = None
    if 77 != model_skel.nbjoints:
        skel_slice = model_skel.get_skel_slice(SOMASkeleton77())

    def _remap(pos77: np.ndarray, rot77: np.ndarray):
        # pos77: (K,77,3), rot77: (K,77,3,3) — SOMA-77. Slice to model skel.
        if skel_slice is not None:
            pos77 = pos77[:, skel_slice]
            rot77 = rot77[:, skel_slice]
        return pos77, rot77

    def _cpu_frames(entries: list) -> torch.Tensor:
        # CPU long tensor — mirrors the chain path's torch.arange (constraints
        # classes build pos_indices/rot_indices on CPU; data on device).
        return torch.tensor(
            [int(e["frame"]) for e in entries], dtype=torch.long,
        )

    objs: list = []

    # ── Full-Body: one FullBodyConstraintSet across all fullbody keyframes ──
    fb = sorted(data.get("fullbody") or [], key=lambda e: e["frame"])
    if fb:
        pos = np.stack([_b64_to_f32(e["jp"]).reshape(77, 3) for e in fb])
        rot = np.stack([_b64_to_f32(e["jr"]).reshape(77, 3, 3) for e in fb])
        pos, rot = _remap(pos, rot)
        objs.append(FullBodyConstraintSet(
            model_skel, _cpu_frames(fb),
            torch.as_tensor(pos, device=device),
            torch.as_tensor(rot, device=device),
            None,  # root2d — the class substitutes the real root (constraints.py:221)
        ))

    # ── End-effectors: one EndEffectorConstraintSet per joint lane ──────────
    ee_by_joint: dict[str, list] = {}
    for e in (data.get("end_effectors") or []):
        ee_by_joint.setdefault(e["joint"], []).append(e)
    for joint, entries in ee_by_joint.items():
        entries.sort(key=lambda e: e["frame"])
        pos = np.stack([_b64_to_f32(e["jp"]).reshape(77, 3) for e in entries])
        rot = np.stack([_b64_to_f32(e["jr"]).reshape(77, 3, 3) for e in entries])
        pos, rot = _remap(pos, rot)
        objs.append(EndEffectorConstraintSet(
            model_skel, _cpu_frames(entries),
            torch.as_tensor(pos, device=device),
            torch.as_tensor(rot, device=device),
            None, joint_names=[joint],
        ))

    # ── 2D Root: one Root2DConstraintSet across all root2d keyframes ────────
    r2 = sorted(data.get("root2d") or [], key=lambda e: e["frame"])
    if r2:
        root2d = torch.tensor(
            [[float(e["x"]), float(e["z"])] for e in r2],
            device=device, dtype=torch.float32,
        )
        objs.append(Root2DConstraintSet(model_skel, _cpu_frames(r2), root2d))

    return objs


def _root_tied_to_one_spot(motion_constraints: str, tol_m: float = 0.02) -> bool:
    """True when every root2d dot sits on ONE spot (within tol_m spread).

    The Motion card's 2D Root lane snapshots the CURRENT root x,z per dot;
    dots authored without dragging land on identical coordinates. Such a set
    expresses "figure stays HERE", not a trajectory to follow.
    """
    try:
        data = json.loads(motion_constraints)
    except json.JSONDecodeError:
        return False
    r2 = data.get("root2d") or []
    if not r2:
        return False
    xs = [float(e["x"]) for e in r2]
    zs = [float(e["z"]) for e in r2]
    spread = max(max(xs) - min(xs), max(zs) - min(zs))
    return spread <= tol_m


# ── Text-encoder lifecycle (the marathon7 lesson, 2026-08-06) ───────────
# kimodo's load_model() instantiates the LLM2Vec text encoder (Llama-3-8B,
# ~21.7GB) internally on EVERY call. The text_encoder_mode node input
# (default "local") plus ComfyUI's execution cache masked this: the loader
# node's output was cached, so the encoder was built once per process and
# reused. The OOM fix's e.reset() drops cached outputs → every fresh load
# stacked a SECOND 21.7GB encoder on the GPU → zero VRAM budget →
# load_models_gpu took the lowvram path (nothing moved) while
# model_patcher still set self.model.device = cuda → "Expected all tensors
# to be on the same device" in diffusion.space_timesteps.
#
# Fix (both halves are required):
#   1. PIN TO CPU — the text_encoder_device node input (default "cpu")
#      makes the encoder live in RAM, never VRAM, so ComfyUI's native
#      model_management always has full VRAM budget for the diffusion
#      model and load_models_gpu never takes the lowvram path.
#   2. OWN THE SINGLETON — the node pack caches the encoder module-level
#      (keyed by variant + device + mode) and passes it back via
#      load_model(..., text_encoder=...) (the documented reuse param:
#      "skips text encoder selection/instantiation entirely"). One 21.7GB
#      instance for the life of the process. Encoding happens on CPU
#      (seconds for one prompt); kimodo_model._generate crosses the
#      result to the GPU via text_feat.to(device).
_TEXT_ENCODER_CACHE: dict[tuple[str, str, str], object] = {}
_TEXT_ENCODER_LOCK = threading.Lock()


# ── Estate location + behavior seats (VENDOR RE-CURE, 2026-11-01) ─────
# COMFYUI EXTENSION NEVER USE ENV VARIABLES. This pack reads and writes
# NO process env for locations or behavior flags — no os.environ.get /
# setdefault / __setitem__, no os.getenv, no putenv. Model homes resolve
# ONLY via ComfyUI's folder_paths categories seated by the endpoint
# statement's extra_model_folders keys (the provisioner regenerates
# extra_model_paths.yaml); the offline/device/mode switches are NODE
# INPUTS on KimodoLoader (defaults preserve the old forced posture),
# never env.
#
# The pip kimodo lib's checkpoint seat is an env-name contract
# (load_model reads CHECKPOINT_DIR through an env helper; the function
# owns no path parameter), and the env-strip re-cure's rebind shim never
# ran — kimodo.model re-exports the load_model FUNCTION over the
# submodule name, so `import kimodo.model.load_model` bound the function
# and the rebind raised AttributeError on the engine. The cure is
# DIRECT: the pack resolves the checkpoint folder itself through the
# avatar folder category (the same _folder_home resolver every other
# home here uses) and instantiates through the lib's own Hydra door
# (config.yaml + instantiate_from_dict), the exact tail load_model()
# would run minus the env seat. The llm2vec encoder conf is likewise
# built from the pack-homed preset, so neither the wrapper's env seats
# nor the lib's env-name selector ever run. Not one env name appears in
# this pack.
_orig_hf_model_info = None


def _folder_home(category: str, decl_home: str) -> str:
    """Resolve a model home: ComfyUI folder_paths category ONLY.

    Estate VENDOR RE-CURE (2026-11-01) — the pack holds no absolute
    model-path literals and reads no location from process env. The
    endpoint statement's extra_model_folders key (seated by the
    provisioner into extra_model_paths.yaml) is the only seat. A home
    found no way refuses LOUD naming the category seat + the DECL
    home — never a silent fallback, never a unit/env seat.

    :param category: the ComfyUI folder_paths category.
    :param decl_home: the estate DECL's lane_homes row (the refusal
        names it so the provisioned tree can be verified).
    """
    try:
        import folder_paths
        hits = folder_paths.get_folder_paths(category)
    except ImportError:
        hits = []
    if hits:
        return hits[0]
    raise RuntimeError(
        f"Kimodo: no {category} home in ComfyUI's folder machinery — "
        f"seat the {category} key in the endpoint statement's "
        f"extra_model_folders and regenerate "
        f"data/runtime/comfyui/extra_model_paths.yaml (node "
        f"--experimental-strip-types "
        f"packages/engines/comfyui-cordis/scripts/provision.ts paths). "
        f"The DECL home is {decl_home}."
    )


def _apply_network_patches(
    *,
    offline: bool,
    text_encoder_device: str,
) -> None:
    """Apply the per-load lib posture: offline hub + folder-homed presets.

    Re-applies on every load so a changed node input takes effect
    without a process restart. No env name is touched anywhere: the
    checkpoint dir is resolved by the pack itself in KimodoLoader.load
    (the lib's env-name checkpoint seat is never consulted), the
    encoder preset rides the llm2vec folder homes + the
    text_encoder_device node input, and the offline posture is a
    huggingface_hub module attribute.
    """
    # The module-level cache (:289) is ASSIGNED below (the offline arm
    # stashes the lib's model_info, the online arm restores it) — without
    # this declaration Python compiles the name function-local and the
    # first read raises UnboundLocalError (the live press red,
    # 2026-11-13 block).
    global _orig_hf_model_info

    # 2026-11-12 (audit 019 C6): the /opt/kimodo sys.path insert that
    # used to live here is DELETED with the retired image layout — the
    # requirements.txt pip leg owns the kimodo import now (see
    # __init__.py's KIMODO SOURCE RESOLUTION note).

    # ── LLM2Vec adapter homes (no process env) ──────────────────────
    # llm2vec_wrapper.py joins its own TEXT_ENCODERS_DIR with the
    # Hydra-provided repo IDs ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-...")
    # to resolve local paths. Rewrite the preset's repo IDs to absolute
    # folder-paths homes BEFORE the encoder conf is built: from_pretrained
    # loads the local tree directly and the wrapper's join leg never
    # fires (it only fires when its own seat is set, which this pack
    # never sets). The llm2vec category (endpoint extra_model_folders)
    # carries the <org>/<repo> layout; without it from_pretrained would
    # try HF Hub and crash with LocalEntryNotFoundError. The device pin
    # rides the preset kwargs too (the wrapper's own device seat stays
    # unset).
    from kimodo.model.load_model import TEXT_ENCODER_PRESETS

    enc_root = _folder_home(
        "llm2vec", "llm2vec/<org>/<repo> adapters "
        "(McGill-NLP/LLM2Vec-Meta-Llama-3-8B-*)")
    for _preset in TEXT_ENCODER_PRESETS.values():
        _kw = _preset.get("kwargs", {})
        for _key in ("base_model_name_or_path", "peft_model_name_or_path"):
            _repo = str(_kw.get(_key, ""))
            if _repo and not os.path.isabs(_repo):
                _kw[_key] = os.path.join(enc_root, _repo)
        _kw["device"] = text_encoder_device

    # ── Offline posture (node input, default True) ──────────────────
    # The mistral tokenizer check (transformers' is_base_mistral /
    # _patch_mistral_regex) phones home during
    # AutoTokenizer.from_pretrained and dies on air-gapped pods. Our
    # model is NOT Mistral, so the offline mock (empty tags →
    # is_base_mistral False) loads. offline=False restores the lib's
    # own model_info and leaves the hub constants to library defaults
    # (an online lane's downloads ride the gateway utility, never this
    # pack).
    try:
        import huggingface_hub
        import huggingface_hub.constants as hf_const
        from types import SimpleNamespace

        if offline:
            hf_const.HF_HUB_OFFLINE = True

            def _offline_model_info(*args, **kwargs):
                return SimpleNamespace(
                    tags=[],
                    id=args[0] if args else kwargs.get("repo_id", ""),
                    config={},
                    siblings=[],
                    sha="",
                    private=False,
                )

            if _orig_hf_model_info is None:
                _orig_hf_model_info = huggingface_hub.model_info
            huggingface_hub.model_info = _offline_model_info
        else:
            hf_const.HF_HUB_OFFLINE = False
            if _orig_hf_model_info is not None:
                huggingface_hub.model_info = _orig_hf_model_info
                _orig_hf_model_info = None
    except ImportError:
        pass


def _text_encoder_conf(text_encoder_mode: str) -> dict:
    """Build the text-encoder conf for the first load (no cached encoder).

    The preset has already been homed + device-pinned by
    _apply_network_patches, so the local conf carries the llm2vec
    folder homes and the node input's device directly — the lib's
    env-name selector (_select_text_encoder_conf) never runs. mode
    "local" (the default) skips the vendor API probe the lib's "auto"
    wastes 5-10s on; mode "auto" keeps the vendor parity semantics —
    one probe of the vendor API service, local on any failure — using
    the lib's own api-conf builder.
    """
    from kimodo.model.load_model import (
        TEXT_ENCODER_PRESETS,
        _build_api_text_encoder_conf,
    )
    from kimodo.model.loading import (
        DEFAULT_TEXT_ENCODER_URL,
        instantiate_from_dict,
    )

    preset = TEXT_ENCODER_PRESETS["llm2vec"]
    local_conf = {"_target_": str(preset["target"]), **preset["kwargs"]}
    if text_encoder_mode != "auto":
        return local_conf
    try:
        api_conf = _build_api_text_encoder_conf(DEFAULT_TEXT_ENCODER_URL)
        probe = instantiate_from_dict(api_conf)
        probe(["healthcheck"])
        return api_conf
    except Exception as error:
        log.info(
            "KimodoLoader: text-encoder API probe failed "
            "(%s: %s) — building the local LLM2Vec encoder",
            type(error).__name__, error,
        )
        return local_conf


# ════════════════════════════════════════════════════════════════════════
# Node 1: Load Kimodo Model  (Rule 1: wrap in comfy.model_patcher.ModelPatcher)
# ════════════════════════════════════════════════════════════════════════
class KimodoLoader:
    """Load a Kimodo diffusion model as a ComfyUI ModelPatcher.

    The model is initialized on the offload device (CPU) and wrapped in
    comfy.model_patcher.ModelPatcher. ComfyUI's memory controller moves
    it to load_device (CUDA) via load_models_gpu when KimodoTextToPose
    runs (see Rule 2).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "default model (leave empty for auto)",
                }),
            },
            "optional": {
                # VENDOR RE-CURE (2026-11-01): the offline/device/mode
                # switches ride node inputs with defaults, never env.
                # "local" skips the phantom text-encoder API probe;
                # offline=True keeps air-gapped pods off the network.
                "text_encoder_device": (["cpu", "cuda"], {
                    "default": "cpu",
                    "tooltip": "LLM2Vec encoder device (cpu keeps VRAM free for the diffusion model)",
                }),
                "text_encoder_mode": (["local", "auto"], {
                    "default": "local",
                    "tooltip": "local builds the LLM2Vec encoder in-process; auto probes the vendor API service first",
                }),
                "offline": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "True blocks huggingface_hub network calls (air-gapped pods); False restores the library defaults",
                }),
            },
        }

    RETURN_TYPES = ("KIMODO_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "Melite/Kimodo"

    def load(
        self,
        model_name: str,
        text_encoder_device: str = "cpu",
        text_encoder_mode: str = "local",
        offline: bool = True,
    ):
        import comfy.model_management as mm
        import comfy.model_patcher as mp
        if text_encoder_mode not in ("local", "auto"):
            raise RuntimeError(
                "KimodoLoader: text_encoder_mode must be 'local' or "
                f"'auto', got {text_encoder_mode!r} — fix the node input."
            )
        if text_encoder_device not in ("cpu", "cuda"):
            raise RuntimeError(
                "KimodoLoader: text_encoder_device must be 'cpu' or "
                f"'cuda', got {text_encoder_device!r} — fix the node input."
            )
        _apply_network_patches(
            offline=offline,
            text_encoder_device=text_encoder_device,
        )

        # ── Free VRAM before Hydra-based loading ──────────────────────────
        # The Hydra instantiate (config.yaml + instantiate_from_dict)
        # instantiates the LLM2Vec text encoder.  The encoder defaults to
        # device="auto" → "cuda" and moves the 16 GB Llama-3 backbone to GPU
        # inside __init__ — BEFORE ComfyUI's load_models_gpu() ever runs.  If
        # a previous brick (e.g. qwen-image-edit) left models cached in
        # ComfyUI's VRAM, those models are still resident and the encoder's
        # allocation OOMs.
        #
        # ComfyUI's automatic VRAM management only fires when ITS OWN loaders
        # call load_models_gpu().  Hydra bypasses that path entirely, so we
        # must explicitly tell ComfyUI to unload everything HERE — giving the
        # Hydra loader a clean VRAM slate.
        mm.unload_all_models()
        mm.soft_empty_cache()

        # NO SILENT DEFAULT (VENDOR RE-CURE): the model name rides the
        # graph from the estate composer (DECL lane_inputs model_path —
        # Kimodo-SOMA-RP-v1.1), resolved under the avatar folder
        # category. A hand-built graph that leaves it empty refuses
        # here, naming the DECL seat + folder category — never a
        # unit/env seat (the pack reads no process env at all).
        variant = model_name.strip()
        if not variant:
            raise RuntimeError(
                "KimodoLoader: model_name is empty — pass the model "
                "directory name (the estate DECL's lane_inputs "
                "model_path, e.g. Kimodo-SOMA-RP-v1.1, resolved as "
                "the avatar folder category's variant subfolder; the "
                "composer always passes it). Model identity rides the "
                "graph, never process env."
            )
        log.info("KimodoLoader: loading %s on CPU (offload device)", variant)
        t0 = time.perf_counter()

        # ── DIRECT checkpoint door (motion cure, 2026-11-01) ─────────────
        # The pinned lib's load_model() owns no path parameter; its only
        # checkpoint seat is an env-name lookup the pack may not feed (the
        # rebind shim that used to redirect it never ran — the package
        # re-exports the load_model FUNCTION over the submodule). The pack
        # resolves the folder itself through the avatar category (the same
        # resolver every other home here uses) and instantiates through the
        # lib's own Hydra door: config.yaml merged with the checkpoint_dir
        # runtime conf — exactly the tail load_model() runs, minus the env
        # seat. resolve=True folds the ${oc.select:checkpoint_dir}
        # interpolations (ckpt_path, stats_path); the runtime key itself is
        # no Kimodo __init__ param and drops after the fold.
        from omegaconf import OmegaConf
        from kimodo.model.loading import instantiate_from_dict

        avatar_root = Path(_folder_home("avatar", "avatar/kimodo"))
        model_path = avatar_root / variant
        if not (model_path / "config.yaml").exists():
            raise RuntimeError(
                f"KimodoLoader: no checkpoint folder for {variant!r} under "
                f"the avatar folder category ({avatar_root}) — config.yaml "
                f"is missing. The variant must name a provisioned model "
                f"directory (the estate DECL's lane_inputs model_path); "
                f"seat the avatar key in the endpoint statement's "
                f"extra_model_folders and regenerate "
                f"data/runtime/comfyui/extra_model_paths.yaml."
            )

        # Encoder reuse: the cached CPU-pinned text encoder rides the
        # instantiate as a placeholder-free direct injection (see the
        # module-level comment for the marathon7 story). None on first load
        # → the pack-built encoder conf instantiates inside the model and we
        # steal it below. Keyed by variant + device + mode so a changed node
        # input builds (and caches) its own instance instead of reusing a
        # stale one.
        cache_key = (variant, text_encoder_device, text_encoder_mode)
        with _TEXT_ENCODER_LOCK:
            encoder = _TEXT_ENCODER_CACHE.get(cache_key)

        runtime_extra = (
            {} if encoder is not None
            else {"text_encoder": _text_encoder_conf(text_encoder_mode)}
        )
        merged = OmegaConf.merge(
            OmegaConf.load(model_path / "config.yaml"),
            OmegaConf.create(
                {"checkpoint_dir": str(model_path), **runtime_extra}
            ),
        )
        model_cfg = OmegaConf.to_container(merged, resolve=True)
        model_cfg.pop("checkpoint_dir", None)
        if encoder is not None:
            # Prevent Hydra from instantiating a second encoder; the cached
            # one is reattached right after (the lib's own reuse tail).
            model_cfg["text_encoder"] = None

        # Rule 1: instantiate on the offload device. NO .to(cuda).
        raw_model = instantiate_from_dict(
            model_cfg, overrides={"device": "cpu"})
        if encoder is not None:
            raw_model.text_encoder = encoder
        resolved = variant

        if encoder is None:
            # First load: the pack-built encoder conf instantiated the
            # encoder inside the model (device pinned by the
            # text_encoder_device node input via the preset rewrite above).
            # Cache it so every later load passes it back — zero
            # instantiation, zero checkpoint load, zero second 21.7GB
            # resident copy.
            with _TEXT_ENCODER_LOCK:
                if cache_key not in _TEXT_ENCODER_CACHE:
                    internal = getattr(raw_model, "text_encoder", None)
                    if internal is not None:
                        _TEXT_ENCODER_CACHE[cache_key] = internal
                        log.info(
                            "KimodoLoader: cached %s text encoder "
                            "(device=%s, ~21.7GB) — later loads reuse it",
                            variant, text_encoder_device,
                        )

        raw_model.eval()

        load_device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        patcher = mp.ModelPatcher(
            model=raw_model,
            load_device=load_device,
            offload_device=offload_device,
        )

        log.info(
            "KimodoLoader: loaded %s in %.1fs (size=%.2fGB)",
            resolved, time.perf_counter() - t0, patcher.model_size() / 1e9,
        )
        return (patcher,)


# ════════════════════════════════════════════════════════════════════════
# Node 2: Kimodo Text to Pose  (Rule 2: load_models_gpu before inference)
# ════════════════════════════════════════════════════════════════════════
class KimodoTextToPose:
    """Run text-to-motion inference via the Kimodo diffusion model.

    Rule 2: ``comfy.model_management.load_models_gpu([model])`` is called
    before any GPU work. ComfyUI calculates free VRAM, evicts inactive
    models via LRU if needed, then moves OUR model's parameters AND ALL
    buffers (including diffusion's betas_base) to load_device atomically.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("KIMODO_MODEL",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                # upstream Kimodo documents 10 s as the maximum
                # generated motion duration PER PROMPT; a longer clip
                # is motion_prompts segments, each <= 10 s
                "duration_sec": ("FLOAT", {
                    "default": DEFAULT_DURATION_SEC, "min": 0.1, "max": 10.0}),
                "diffusion_steps": ("INT", {
                    "default": DEFAULT_DIFFUSION_STEPS, "min": 1, "max": 1000}),
                "num_samples": ("INT", {
                    "default": DEFAULT_NUM_SAMPLES, "min": 1, "max": 8}),
                "num_transition_frames": ("INT", {
                    "default": DEFAULT_NUM_TRANSITION_FRAMES, "min": 0, "max": 300}),
                "cfg_type": (["separated", "regular", "nocfg"], {
                    "default": DEFAULT_CFG_TYPE}),
                "cfg_weight_a": ("FLOAT", {
                    "default": DEFAULT_CFG_WEIGHT[0], "min": 0.0, "max": 20.0}),
                "cfg_weight_b": ("FLOAT", {
                    "default": DEFAULT_CFG_WEIGHT[1], "min": 0.0, "max": 20.0}),
                # Chain mode: bare ComfyUI input filename of a previous
                # motion NPZ. When set, the new motion is generated
                # conditioned on the previous clip's final pose (frame-0
                # constraints) and the OUTPUT is the MERGED chain.
                "prev_motion_npz": ("STRING", {"default": "", "multiline": False}),
                # User-authored timeline constraints (the motion card's bar).
                # JSON serialized by web/editor/.../constraints.ts:
                #   { fullbody?: [{frame, jp, jr}, ...],      # jp/jr = b64 '<f4'
                #     end_effectors?: [{frame, joint, jp, jr}, ...],
                #     root2d?: [{frame, x, z}, ...] }
                # jp = 77*3 float32 (SOMA-77, meters), jr = 77*9 (row-major
                # 3x3). Built into FullBodyConstraintSet /
                # EndEffectorConstraintSet / Root2DConstraintSet and fed to
                # raw()'s constraint_lst — the motion equivalent of the
                # Kimodo demo's compute_model_constraints_lst (generation.py).
                # When present AND non-empty, this takes precedence over the
                # chain-anchor path (mutually exclusive: author constraints
                # OR chain-repose, not both).
                "motion_constraints": ("STRING", {"default": "", "multiline": True}),
                # User-authored MULTI-PROMPT timeline (the motion card's prompt
                # lane). JSON serialized by web/editor/.../constraints.ts:
                #   [{"text": "walk forward", "duration_sec": 2.0}, {"text": "wave", "duration_sec": 1.5}, ...]
                # Sorted ascending by timeline start_frame; durations are per-
                # segment seconds (demo ui.py:1646-1659 convention). When
                # present with >=2 entries, this OVERRIDES single-prompt mode:
                # raw() receives the full prompts list with multi_prompt=True
                # so the model composes a multi-segment motion with transitions
                # (kimodo_model.py:380 __call__ → _generate iterates the list).
                # This is the demo's PRIMARY authoring surface — each prompt
                # segment is a motion "card" stitched into one clip.
                "motion_prompts": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "KIMODO_RESULT")
    RETURN_NAMES = ("npz_path", "result")
    FUNCTION = "generate"
    CATEGORY = "Melite/Kimodo"
    OUTPUT_NODE = True

    def generate(
        self,
        model,
        prompt: str,
        seed: int = 42,
        duration_sec: float = DEFAULT_DURATION_SEC,
        diffusion_steps: int = DEFAULT_DIFFUSION_STEPS,
        num_samples: int = DEFAULT_NUM_SAMPLES,
        num_transition_frames: int = DEFAULT_NUM_TRANSITION_FRAMES,
        cfg_type: str = DEFAULT_CFG_TYPE,
        cfg_weight_a: float = DEFAULT_CFG_WEIGHT[0],
        cfg_weight_b: float = DEFAULT_CFG_WEIGHT[1],
        prev_motion_npz: str = "",
        motion_constraints: str = "",
        motion_prompts: str = "",
    ):
        if not prompt or not prompt.strip():
            raise RuntimeError("KimodoTextToPose: prompt is required")
        if len(prompt) > 2000:
            raise RuntimeError(
                f"KimodoTextToPose: prompt length {len(prompt)} exceeds 2000 char limit"
            )

        # ── Multi-prompt timeline parse (the demo's primary authoring path) ──
        # motion_prompts is a JSON list of {text, duration_sec} segments
        # authored on the timeline's prompt lane. When >=2 valid entries are
        # present, this takes precedence over the single `prompt` param: the
        # model composes a multi-segment motion with transitions
        # (kimodo_model.py _generate iterates the list). Falls back to the
        # single-prompt path on any parse failure or single-entry list, so a
        # malformed payload never crashes (hard-fail loud below only on
        # explicitly-present-but-invalid shape). Mirrors demo/ui.py:1646-1659.
        import json as _json
        prompts_list: list[str] = [prompt]
        multi_prompt_durations_sec: list[float] = []
        multi_prompt_segments = 0
        if motion_prompts and motion_prompts.strip():
            try:
                parsed = _json.loads(motion_prompts)
            except _json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"KimodoTextToPose: motion_prompts is not valid JSON ({exc}). "
                    f"Expected a list of {{text, duration_sec}} objects."
                ) from exc
            if not isinstance(parsed, list):
                raise RuntimeError(
                    "KimodoTextToPose: motion_prompts must be a JSON list, "
                    f"got {type(parsed).__name__}"
                )
            cleaned: list[tuple[str, float]] = []
            for seg in parsed:
                if not isinstance(seg, dict):
                    continue
                text = seg.get("text", "")
                dur = seg.get("duration_sec", 0)
                if not isinstance(text, str) or not text.strip():
                    continue
                try:
                    dur_f = float(dur)
                except (TypeError, ValueError):
                    continue
                if dur_f <= 0:
                    continue
                cleaned.append((text.strip(), dur_f))
            if len(cleaned) >= 2:
                prompts_list = [t for t, _ in cleaned]
                multi_prompt_durations_sec = [d for _, d in cleaned]
                multi_prompt_segments = len(cleaned)
                log.info(
                    "KimodoTextToPose: %d multi-prompt segments from motion_prompts "
                    "(overriding single prompt, multi_prompt=True)",
                    multi_prompt_segments,
                )

        # Rule 2: load the model onto load_device via ComfyUI's memory mgr.
        # This is the contract that prevents the device-mismatch bug.
        import comfy.model_management as mm
        mm.load_models_gpu([model])

        raw = model.model            # underlying nn.Module
        device = model.load_device   # source of truth (Rule 4)
        fps = getattr(raw, "fps", DEFAULT_FPS)
        if multi_prompt_segments >= 2:
            # Per-segment frame counts from the parsed motion_prompts durations.
            num_frames = [max(1, int(round(d * fps))) for d in multi_prompt_durations_sec]
        else:
            num_frames = [max(1, int(round(duration_sec * fps)))]

        # CFG kwargs (mirrors media/motion/kimodo_pose.py:265-273)
        cfg_kwargs = {}
        if cfg_type and cfg_type != "nocfg":
            cfg_kwargs["cfg_type"] = cfg_type
            cfg_kwargs["cfg_weight"] = [cfg_weight_a, cfg_weight_b]

        # G1 models skip post-processing (per upstream generate.py)
        resolved = getattr(raw, "name", "") or ""
        use_postprocess = "g1" not in resolved.lower()

        # ── Chain mode (prev_motion_npz set) ─────────────────────────────
        # Condition the new motion on the previous clip's final pose via
        # frame-0 constraints — the motion equivalent of H3's
        # first_frame/last_frame. The construction mirrors the model's own
        # _multiprompt transition logic (kimodo_model.py:205-225): a
        # FullBody constraint anchors positions/root, and a separate
        # EndEffector constraint locks hand/feet ROTATIONS (fullbody
        # conditioning ignores rotations — constraints.py:246). Enforcement
        # is a hard per-timestep mask replacement (twostage_denoiser), so
        # the new clip genuinely starts where the previous one ended.
        chained = bool(prev_motion_npz)
        constraint_lst: list[list[object]] = []
        prev_data: dict[str, np.ndarray] = {}
        if chained:
            import folder_paths
            from kimodo.constraints import (
                EndEffectorConstraintSet,
                FullBodyConstraintSet,
            )

            prev_path = os.path.join(
                folder_paths.get_input_directory(), prev_motion_npz)
            with np.load(prev_path) as z:
                prev_data = {k: np.asarray(z[k]) for k in z.files}
            pj = prev_data.get("posed_joints")
            gr = prev_data.get("global_rot_mats")
            if pj is None or gr is None:
                raise RuntimeError(
                    f"KimodoTextToPose: chained NPZ {prev_motion_npz!r} is "
                    f"missing posed_joints/global_rot_mats — cannot "
                    f"condition on it"
                )
            if pj.ndim != 3 or gr.ndim != 4 or pj.shape[2] != 3:
                raise RuntimeError(
                    f"KimodoTextToPose: chained NPZ {prev_motion_npz!r} has "
                    f"unexpected shapes posed_joints={pj.shape} "
                    f"global_rot_mats={gr.shape}"
                )

            # Skeleton remap: chained NPZs are SOMASkeleton77-ordered (our
            # output contract); the model denoises on its internal skeleton
            # (SOMASkeleton30). Same remap as the demo bridge
            # (demo/generation.py:38,73-75). Guard with nbjoints so a
            # future 30-joint input passes through untouched.
            model_skel = raw.skeleton
            if pj.shape[1] != model_skel.nbjoints:
                from kimodo.skeleton.definitions import SOMASkeleton77
                skel_slice = model_skel.get_skel_slice(SOMASkeleton77())
                pj = pj[:, skel_slice]
                gr = gr[:, skel_slice]

            # K transition frames: the new clip's first K frames are
            # hard-anchored to the prev clip's last K poses. Clamp to the
            # prev clip's length.
            K = int(num_transition_frames)
            K = max(0, min(K, len(pj)))
            if K < 1:
                raise ValueError(
                    "KimodoTextToPose: chaining requires num_transition_"
                    f"frames >= 1 (got {num_transition_frames})"
                )
            # CPU indices — mirrors _multiprompt exactly (kimodo_model.py:
            # torch.arange(num_transition_frames) has no device arg). The
            # constraint classes build pos_indices/rot_indices on CPU
            # (constraints.py:351-352); create_pairs would hit a CPU/cuda
            # mismatch in torch.stack otherwise (proven 2026-08-08 gate 3:
            # "Expected all tensors to be on the same device... wrapper_
            # CUDA_cat"). The conditioning machinery moves indices to the
            # model device later (create_conditions_from_constraints_batched
            # passes device=...). Data tensors stay on the model device,
            # exactly like _multiprompt's last_output.
            frame_idx = torch.arange(K)
            # clone(): as_tensor may alias the prev NPZ's array (CPU path),
            # and pos is translated in place below — the merge reads the
            # prev arrays verbatim, so the anchor window must not mutate
            # them.
            pos = torch.as_tensor(pj[-K:], device=device).clone()  # (K, J, 3)
            rot = torch.as_tensor(gr[-K:], device=device)      # (K, J, 3, 3)
            smooth = prev_data.get("smooth_root_pos")
            # Root x/z continuity; None → the constraint class substitutes
            # the real root of the anchored frames (constraints.py:221-223).
            root2d = None
            if smooth is not None and smooth.ndim == 2:
                root2d = torch.as_tensor(smooth[-K:, [0, 2]], device=device)

            # Canonicalize the anchor window to the model's motion prior —
            # the single-prompt mirror of _multiprompt's
            # translate_2d(observed_motion, -last_smooth_root_2d) round
            # trip (kimodo_model.py:249-250,283-284). The denoiser's prior
            # generates free frames around the canonical root origin; a
            # world-frame anchor meters away makes the first free frame pop
            # back to it (proven 2026-08-08 gate 3: 1.36 m body-wide
            # junction jump). Anchor at root ≈ 0 for the denoiser, restore
            # the world root on the output below.
            if root2d is not None:
                chain_root0 = root2d[0].clone()
                root2d = root2d - chain_root0
            else:
                chain_root0 = pos[0, model_skel.root_idx, [0, 2]].clone()
            pos[..., 0] -= chain_root0[0]
            pos[..., 2] -= chain_root0[1]
            chain_root0_np = chain_root0.detach().cpu().numpy()

            full = FullBodyConstraintSet(
                model_skel, frame_idx, pos, rot, root2d)
            ee = EndEffectorConstraintSet(
                model_skel, frame_idx, pos, rot, root2d,
                joint_names=["LeftHand", "RightHand", "LeftFoot", "RightFoot"],
            )
            constraint_lst = [[full, ee] for _ in range(num_samples)]

            # The stated duration_sec is NEW content: request K extra
            # transition frames that are merged out of the output below.
            num_frames = [num_frames[0] + K]
            log.info(
                "KimodoTextToPose: chained from %s (%d prev frames, K=%d) — "
                "generating %d frames total",
                prev_motion_npz, len(pj), K, num_frames[0],
            )

        # ── User-authored constraints (the motion card's timeline bar) ──────
        # When the editor's bar produced keyframes, they arrive here as JSON
        # (web/editor/.../constraints.ts serializeConstraints). Each lane's
        # keyframes become constraint objects the SAME way the Kimodo demo's
        # compute_model_constraints_lst builds them (generation.py:24-111):
        #   Full-Body  lane → FullBodyConstraintSet (all 77 joints)
        #   L/R Hand   lane → EndEffectorConstraintSet(joint_names=[that joint])
        #   L/R Foot   lane → EndEffectorConstraintSet(joint_names=[that joint])
        #   2D Root    lane → Root2DConstraintSet (root x,z trajectory)
        # User constraints take PRECEDENCE over the chain-anchor path — they
        # are mutually exclusive in practice (you author constraints for a
        # fresh generation; you chain-repose from an edited pose). This is
        # what makes the bar DRIVE generation instead of rendering inert.
        inplace_bake = False
        if motion_constraints and motion_constraints.strip():
            user_objs = _build_user_constraints(
                motion_constraints, raw.skeleton, device,
            )
            # -- In-place tie (2026-08-24) -------------------------------
            # root2d dots that all sit on one spot mean "figure stays here",
            # NOT a path to trace. Enforcing them during sampling pins the
            # pelvis for the whole clip, and the prior answers with a frozen
            # stance: leg articulation collapses ~307 -> ~17 cm/s vs the same
            # seed generated free (proven runs pmt7bkbevi9w9or vs
            # pmt7bxt6oam9c07). Demo parity instead: generate FREE (full
            # gait) and bake the treadmill onto the RESULT below -- feet keep
            # moving, the figure never leaves its spot. Non-degenerate root
            # paths still steer sampling as real Root2D constraints.
            if _root_tied_to_one_spot(motion_constraints):
                n_r2d = sum(
                    1 for o in user_objs if getattr(o, "name", "") == "root2d"
                )
                user_objs = [
                    o for o in user_objs
                    if getattr(o, "name", "") != "root2d"
                ]
                inplace_bake = True
                log.info(
                    "KimodoTextToPose: root2d ties the figure to one spot "
                    "(%d constraint set(s), <=2cm spread) -- dropping the "
                    "sampling pin, generating free, baking in-place travel "
                    "removal",
                    n_r2d,
                )
            if user_objs:
                # FLAT list of constraint objects — the demo's shape
                # (generation.py compute_model_constraints_lst returns a flat
                # list and passes it straight to raw()). Standalone runs go
                # through _multiprompt, which calls constraint.crop_move(...) on
                # EACH element (kimodo_model.py:177) — a per-sample nested
                # [[obj], ...] crashed there with "'list' object has no
                # attribute 'crop_move'" (proven 2026-08-24 run
                # pmt7bkimo6y21c9). create_conditions_from_constraints_batched
                # accepts a shared flat list in BOTH paths (base.py:275 — it
                # repeats it across samples), and post_process_motion treats a
                # flat list as shared masks — which is exactly right: user
                # authoring applies identically to every sample.
                constraint_lst = list(user_objs)
                # User constraints are NOT a chain merge — generate standalone
                # with the constraints applied (multi_prompt stays True, no
                # transition-frame merge).
                log.info(
                    "KimodoTextToPose: %d user constraint objects from "
                    "motion_constraints (fullbody/end_effectors/root2d) — "
                    "overriding chain path",
                    len(user_objs),
                )

        # Seed via Kimodo's seed_everything (seeds python random + numpy +
        # torch). Falls back to torch.manual_seed if unavailable.
        try:
            from kimodo.tools import seed_everything
            seed_everything(int(seed))
        except Exception as e:
            log.warning("seed_everything failed (%s) — using torch.manual_seed", e)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

        log.info(
            "KimodoTextToPose: diffusion (%d frames, seed=%s, steps=%d, "
            "samples=%d) on %s",
            num_frames[0], seed, diffusion_steps, num_samples, device,
        )
        t0 = time.perf_counter()

        # Identical call shape to the prior service (proven to produce
        # valid SOMA-77 motion data when the model is correctly loaded).
        # progress_bar=_InterruptibleBar(): the Freeze hook — per-step
        # ComfyUI interrupt check (see the class docstring).
        with torch.inference_mode():
            output = raw(
                prompts_list if multi_prompt_segments >= 2 else [prompt],
                num_frames,
                constraint_lst=constraint_lst,
                num_denoising_steps=diffusion_steps,
                num_samples=num_samples,
                multi_prompt=(multi_prompt_segments >= 2) or (not chained),
                num_transition_frames=num_transition_frames,
                post_processing=use_postprocess,
                return_numpy=True,
                progress_bar=_InterruptibleBar(),
                **cfg_kwargs,
            )
        infer_s = time.perf_counter() - t0

        posed_joints = np.asarray(output["posed_joints"])
        global_rot_mats = output.get("global_rot_mats")
        if global_rot_mats is not None:
            global_rot_mats = np.asarray(global_rot_mats)
        # Normalize sample-0 (num_samples>1 produces batched output).
        if posed_joints.ndim == 4:  # (N, T, J, 3)
            posed_joints = posed_joints[0]
        if global_rot_mats is not None and global_rot_mats.ndim == 5:  # (N, T, J, 3, 3)
            global_rot_mats = global_rot_mats[0]

        npz_data = {"posed_joints": posed_joints.astype(np.float32)}
        if global_rot_mats is not None:
            npz_data["global_rot_mats"] = global_rot_mats.astype(np.float32)
        for k, v in output.items():
            if k in ("posed_joints", "global_rot_mats"):
                continue
            try:
                arr = np.asarray(v)
                if arr.ndim > 0 and arr.shape[0] == num_samples:
                    arr = arr[0]
                npz_data[k] = arr
            except Exception:
                pass

        # ── World-frame restore (chained) ───────────────────────────────
        # The model generated in the canonical anchor frame; translate the
        # output back by the window origin — the translate_2d(motion,
        # last_smooth_root_2d) half of _multiprompt's round trip. Only
        # root-dependent position keys move; rotations, contacts and
        # headings are translation-invariant.
        if chained:
            for k in ("posed_joints", "root_positions", "smooth_root_pos"):
                arr = npz_data.get(k)
                if arr is not None and arr.ndim >= 2:
                    arr[..., 0] += chain_root0_np[0]
                    arr[..., 2] += chain_root0_np[1]

        # ── Chain merge ─────────────────────────────────────────────────
        # B's first K frames are hard-anchored to A's ending pose, so the
        # merged chain = A's frames verbatim + B's UNCONSTRAINED frames
        # (the K transition frames are dropped, exactly like _multiprompt's
        # motion_with_transition[:, num_transition_frames:]). All
        # frame-indexed output keys share the same timeline — a shape[0]
        # equal to the generated total marks them.
        if chained:
            total = posed_joints.shape[0]
            new_len = total - K
            for k, arr in npz_data.items():
                if arr.ndim < 1 or arr.shape[0] != total:
                    continue
                prev_arr = prev_data.get(k)
                if (prev_arr is not None and prev_arr.ndim == arr.ndim
                        and prev_arr.shape[1:] == arr.shape[1:]):
                    npz_data[k] = np.concatenate([prev_arr, arr[K:]], axis=0)
                else:
                    # Key absent in the prev clip (or shape drift): keep
                    # the new clip alone, transition frames dropped.
                    npz_data[k] = arr[K:]
            posed_joints = npz_data["posed_joints"]
            if "global_rot_mats" in npz_data:
                global_rot_mats = npz_data["global_rot_mats"]
            log.info(
                "KimodoTextToPose: chained merge → %d frames "
                "(prev %d + new %d)",
                posed_joints.shape[0], len(pj), new_len,
            )

        # -- In-place bake (tied-root demo parity, 2026-08-24) ------------
        # Generation ran FREE (the constant-spot root2d pin was dropped):
        # remove the root's horizontal travel from the RESULT by shifting
        # every frame back by its root XZ displacement from frame 0. Y is
        # kept (vertical bob); rotations, contacts and headings are
        # translation-invariant. Feet keep their full articulation while the
        # figure stays at its spot -- the Kimodo demo's inPlacePlayback
        # view, baked into the data.
        if inplace_bake:
            base = npz_data.get("posed_joints")
            if base is None or base.ndim != 3 or base.shape[-1] != 3:
                raise RuntimeError(
                    "KimodoTextToPose: in-place bake requested but "
                    "posed_joints has unexpected shape "
                    f"{None if base is None else list(base.shape)}"
                )
            root0 = base[:, 0, :].copy()
            disp = root0 - root0[0:1, :]
            disp[:, 1] = 0.0  # keep vertical bob
            total_frames = base.shape[0]
            travel_before = float(np.linalg.norm(np.diff(root0[:, [0, 2]], axis=0), axis=1).sum())
            for k in ("posed_joints", "root_positions", "smooth_root_pos"):
                arr = npz_data.get(k)
                if (
                    arr is not None and arr.ndim >= 2
                    and arr.shape[-1] == 3 and arr.shape[0] == total_frames
                ):
                    npz_data[k] = arr - disp[:total_frames, None, :]
            after = npz_data["posed_joints"][:, 0, :]
            net_after = float(np.linalg.norm(after[-1, [0, 2]] - after[0, [0, 2]]))
            log.info(
                "KimodoTextToPose: in-place bake applied — removed %.2fm of "
                "root XZ path length; net travel now %.2fm",
                travel_before, net_after,
            )

        # Write NPZ to ComfyUI's output dir (standard pattern, matches
        # Trellis2ExportTrimesh etc -- surfaces via ui.three_model).
        import folder_paths
        out_dir = folder_paths.get_output_directory()
        out_name = f"kimodo_pose_{int(time.time() * 1000)}_{seed}.npz"
        out_path = os.path.join(out_dir, out_name)
        np.savez_compressed(out_path, **npz_data)

        frame_count = int(posed_joints.shape[0])
        log.info(
            "KimodoTextToPose: done — %d frames (%.1fs motion) in %.1fs -> %s",
            frame_count, frame_count / fps, infer_s, out_name,
        )

        result_meta = {
            "motion_shape": list(posed_joints.shape),
            "fps": float(fps),
            "prompt": prompt,
            "seed": seed,
            "inference_seconds": infer_s,
        }
        if multi_prompt_segments >= 2:
            result_meta["multi_prompt_segments"] = multi_prompt_segments
            result_meta["prompts"] = prompts_list
        if chained:
            result_meta["chained_from"] = prev_motion_npz
            result_meta["prev_frames"] = int(len(pj))
        if inplace_bake:
            result_meta["in_place"] = True

        return {
            "ui": {"three_model": [{
                "filename": out_name, "subfolder": "", "type": "output",
            }]},
            "result": (
                out_path,
                result_meta,
            ),
        }


# ════════════════════════════════════════════════════════════════════════
# Registration
# ════════════════════════════════════════════════════════════════════════
NODE_CLASS_MAPPINGS = {
    "KimodoLoader": KimodoLoader,
    "KimodoTextToPose": KimodoTextToPose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KimodoLoader": "Load Kimodo Model",
    "KimodoTextToPose": "Kimodo Text to Pose",
}
