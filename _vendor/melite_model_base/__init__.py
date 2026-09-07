"""Shared base class for registering diffusers-style pipelines with ComfyUI's
``comfy.model_management`` VRAM tracking.

PROBLEM
-------
``melite-trellis-nodes`` (and anigen, latentsync, kimodo, ...) wrap external
diffusers-style pipelines as module-level globals::

    _pipeline_cache: dict[str, object] = {}
    pipeline = TrellisPipeline.from_pretrained(...)
    pipeline.cuda()
    _pipeline_cache[ckpt] = pipeline

These pipelines are NOT ``ModelPatcher`` objects, so ComfyUI's
``current_loaded_models`` list doesn't know they exist. When ComfyUI's
``load_models_gpu`` runs to load a new model, it calls ``free_memory`` to
evict old tracked models — but our pipelines are INVISIBLE to that loop.
They squat on VRAM until we manually clear the cache.

SOLUTION
--------
``PipelinePatcher`` wraps any object with ``.to(device)`` (and sub-modules
that expose ``.parameters()``) so it appears in
``current_loaded_models`` just like a ComfyUI-tracked model. ComfyUI can
then:

  * Correctly account for the pipeline's VRAM in ``get_free_memory``.
  * Choose to evict it via ``free_memory`` when VRAM is needed for another
    model (moves it to CPU automatically).
  * Report it in ``/system_stats`` for observability.

The patcher is kept alive by the caller (typically held in a
``_pipeline_cache`` dict). When the caller drops the reference, Python
GC's the patcher, ``LoadedModel.is_dead()`` returns True, and ComfyUI's
``cleanup_models()`` removes the entry.

INSTALLATION
------------
This is a regular Python package, NOT a ComfyUI custom node. ComfyUI loads
custom nodes in filesystem-dependent order which races with cross-package
imports, so depending on an installed copy alone would be fragile.

Canonical source: ``packages/melite-model-base/src/melite_model_base/``. Every
consuming node pack carries a synced copy under ``<pack>/_vendor/`` and
falls back to it when no installed ``melite_model_base`` is importable —
making each pack standalone (drop it into ``custom_nodes/`` and it works;
no mounts, no PYTHONPATH). A ``pip install melite-model-base`` always wins
over the vendored copy. Copies are kept byte-identical by
``scripts/sync_melite_model_base.py`` (``--check`` mode runs in CI).

Lifecycle methods (called by ComfyUI):

  * ``partially_load(device, _)``   → ``pipeline.to(device)`` (full move)
  * ``partially_unload(device, _)`` → no-op (we don't partial-unload)
  * ``detach(_)``                   → ``pipeline.to(offload_device)`` (CPU)
"""
from __future__ import annotations

import gc
import logging
import threading
import weakref
from typing import Any, Optional

import torch

log = logging.getLogger(__name__)


def _mm():
    """Lazy import of comfy.model_management. Fails gracefully outside ComfyUI."""
    import comfy.model_management
    return comfy.model_management


LOAD_DEVICE: torch.device = (
    # torch.device("cuda") (no index) is NOT equal to torch.device("cuda:0")
    # per PyTorch's __eq__, even though they reference the same device.
    # Pin to cuda:0 so device comparisons against parameter .device (which
    # reports cuda:0 after .to("cuda")) are consistent.
    torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
)
OFFLOAD_DEVICE: torch.device = torch.device("cpu")


class PipelinePatcher:
    """Wrap a diffusers-like pipeline so ComfyUI tracks its VRAM.

    Implements the subset of the ``ModelPatcher`` interface that
    ``load_models_gpu`` and ``LoadedModel`` actually call. Everything
    related to weight patching / LoRA / hooks is intentionally absent —
    our pipelines are loaded wholesale, not patched.

    The patcher must be kept alive by the caller (e.g. held in a module
    cache). ``LoadedModel`` holds a weakref, so when the caller drops the
    last strong reference, Python GC's the patcher and ComfyUI's
    ``cleanup_models`` removes the entry.
    """

    # ComfyUI LoadedModel checks ``model.parent`` — must exist, None = no parent
    parent: Optional["PipelinePatcher"] = None

    # Subclasses can override to identify the pipeline type in logs / health
    pipeline_type: str = "pipeline"

    def __init__(
        self,
        pipeline: Any,
        name: str = "pipeline",
        load_device: Optional[torch.device] = None,
        offload_device: Optional[torch.device] = None,
        size_bytes: Optional[int] = None,
    ):
        # `pipeline` becomes ``self.model`` — LoadedModel reads this and
        # weakrefs it. MUST be the actual diffusers pipeline object.
        self.model = pipeline
        self.name = name
        self.load_device = load_device or LOAD_DEVICE
        self.offload_device = offload_device or OFFLOAD_DEVICE
        self._size_bytes = (
            size_bytes if size_bytes is not None
            else self._measure_size(pipeline)
        )
        self._dtype = torch.float16
        # ComfyUI may set this via calculate_weight callbacks; we ignore it
        # but the attribute must exist.
        self.calculated: list = []

    # ─── Size accounting ─────────────────────────────────────────────────
    @staticmethod
    def _iter_sub_modules(pipeline: Any):
        """Yield nn.Module attributes of a diffusers-style pipeline.

        Diffusers ``Pipeline`` objects aren't nn.Modules themselves but
        hold sub-modules (unet, vae, text_encoder, feature_extractor, ...)
        as instance attributes. We walk ``vars(pipeline)`` and yield each
        nn.Module we find, deduping by id() so shared modules aren't
        double-counted.
        """
        if not hasattr(pipeline, "__dict__"):
            return
        seen = set()
        for v in vars(pipeline).values():
            candidates = []
            if isinstance(v, torch.nn.Module):
                candidates.append(v)
            elif isinstance(v, (list, tuple)):
                candidates.extend(x for x in v if isinstance(x, torch.nn.Module))
            elif isinstance(v, dict):
                candidates.extend(
                    x for x in v.values() if isinstance(x, torch.nn.Module)
                )
            for m in candidates:
                if id(m) not in seen:
                    seen.add(id(m))
                    yield m

    @classmethod
    def _measure_size(cls, pipeline: Any) -> int:
        """Sum parameter + buffer bytes.

        Works for: (a) nn.Module instances (calls parameters()/buffers()
        with recurse=True directly), (b) diffusers-style pipelines (walks
        sub-module attrs and recurses within each), (c) any object exposing
        either interface.

        Avoids double-counting: if ``pipeline`` is itself an nn.Module, its
        ``parameters(recurse=True)`` already covers every nested sub-module —
        so we do NOT also call ``walk`` on the iterated sub-modules. The
        sub-module iteration only kicks in for diffusers-style pipelines
        that aren't nn.Modules themselves.
        """
        total = 0
        seen_mods = set()

        def walk(mod: torch.nn.Module):
            nonlocal total
            if id(mod) in seen_mods:
                return
            seen_mods.add(id(mod))
            try:
                for p in mod.parameters(recurse=True):
                    total += p.nelement() * p.element_size()
            except (AttributeError, RuntimeError):
                pass
            try:
                for b in mod.buffers(recurse=True):
                    total += b.nelement() * b.element_size()
            except (AttributeError, RuntimeError):
                pass

        if isinstance(pipeline, torch.nn.Module):
            # pipeline IS an nn.Module: parameters(recurse=True) covers all
            # nested children — don't double-count via sub-module iteration.
            walk(pipeline)
        else:
            # diffusers-style pipeline: walk each sub-module separately.
            for sub in cls._iter_sub_modules(pipeline):
                walk(sub)
        return total

    def model_size(self) -> int:
        return self._size_bytes

    def loaded_size(self) -> int:
        """Bytes of THIS model currently resident on load_device.

        For diffusers pipelines with sequential CPU offloading (TRELLIS
        ``low_vram=True``, diffusers ``device_map='sequential'`` etc), the
        full ``model_size()`` overstates VRAM use because most sub-modules
        live on CPU between stages. ComfyUI uses this to decide how much
        to evict, so we must report the ACTUAL GPU-resident bytes —
        walking sub-modules and counting those on load_device.
        """
        return self._measure_size_on_device(self.model, self.load_device)

    @classmethod
    def _measure_size_on_device(cls, pipeline: Any, device: torch.device) -> int:
        """Sum parameter + buffer bytes whose .device matches ``device``."""
        total = 0
        seen = set()

        def walk(mod: torch.nn.Module):
            nonlocal total
            if id(mod) in seen:
                return
            seen.add(id(mod))
            try:
                for p in mod.parameters(recurse=False):
                    if cls._device_eq(p.device, device):
                        total += p.nelement() * p.element_size()
            except (AttributeError, RuntimeError):
                pass
            try:
                for b in mod.buffers(recurse=False):
                    if cls._device_eq(b.device, device):
                        total += b.nelement() * b.element_size()
            except (AttributeError, RuntimeError):
                pass
            for child in mod.children():
                walk(child)

        if isinstance(pipeline, torch.nn.Module):
            walk(pipeline)
        for sub in cls._iter_sub_modules(pipeline):
            walk(sub)
        return total

    @staticmethod
    def _device_eq(a: torch.device, b: torch.device) -> bool:
        """Device equality that treats ``cuda`` as ``cuda:0``.

        PyTorch's ``__eq__`` returns False for ``torch.device("cuda")`` vs
        ``torch.device("cuda:0")`` even though they reference the same
        physical device. Parameters moved via ``.to("cuda")`` end up with
        ``device.type="cuda", device.index=0`` (or the current device),
        but our ``load_device`` defaults to ``torch.device("cuda")`` (no
        index). Normalize by treating a None index as 0 for cuda.
        """
        if a.type != b.type:
            return False
        if a.type == "cpu":
            return True
        # For cuda/etc, treat None index as 0 (CUDA default device).
        return (a.index or 0) == (b.index or 0)

    def current_loaded_device(self) -> torch.device:
        """Return the device of the first parameter we find.

        For diffusers pipelines with sub-module offloading, this reflects
        where the FIRST sub-module lives (which may be CPU even when other
        sub-modules are on GPU). Used by ComfyUI for accounting decisions;
        the precise per-sub-module accounting is in ``loaded_size``.
        """
        pipeline = self.model
        if isinstance(pipeline, torch.nn.Module):
            try:
                return next(pipeline.parameters()).device
            except StopIteration:
                pass
        for sub in self._iter_sub_modules(pipeline):
            try:
                return next(sub.parameters()).device
            except StopIteration:
                continue
        return self.offload_device

    def model_loaded_memory(self) -> int:
        return self.loaded_size()

    def model_offloaded_memory(self) -> int:
        return self.model_size() - self.loaded_size()

    def model_memory_required(self, device: torch.device) -> int:
        """Bytes that would need to be moved to ``device`` to fully load.

        For pipelines already partially on ``device``, this is just the
        offloaded portion. For pipelines not on ``device`` at all, it's
        the full size.
        """
        if device == self.load_device:
            return self.model_offloaded_memory()
        return self.model_size()

    # ─── Lifecycle hooks (called by LoadedModel / load_models_gpu) ───────
    def is_dynamic(self) -> bool:
        return False

    def is_clone(self, other: Any) -> bool:
        return other is self

    def lowvram_patch_counter(self) -> int:
        return 0

    def model_patches_models(self) -> list:
        return []

    def model_dtype(self) -> torch.dtype:
        return self._dtype

    def model_patches_to(self, target) -> None:
        """Called with a device OR a dtype by LoadedModel.model_load.

        We don't patch weights — just remember the dtype if it's a dtype.
        Device moves happen via ``partially_load``.
        """
        if isinstance(target, torch.dtype):
            self._dtype = target

    def partially_load(
        self,
        device: torch.device,
        extra_memory: int,
        force_patch_weights: bool = False,
    ) -> int:
        """Move pipeline to ``device``. Returns bytes loaded (always full)."""
        if self.current_loaded_device() == device:
            return 0
        try:
            self.model.to(device)
        except Exception as e:
            log.error(
                "[PipelinePatcher] %s.to(%s) failed: %s",
                self.name, device, e,
            )
            raise
        return self._size_bytes

    def partially_unload(
        self,
        device: torch.device,
        memory_to_free: int,
    ) -> int:
        """Partial unload — not supported. Return 0 so ComfyUI falls back
        to ``detach()`` for full unload."""
        return 0

    def detach(self, unpatch_weights: bool = True) -> None:
        """Full unload — move pipeline to offload_device (CPU)."""
        try:
            self.model.to(self.offload_device)
        except Exception as e:
            log.warning(
                "[PipelinePatcher] detach move failed for %s: %s",
                self.name, e,
            )
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def should_reload_model(self, force_patch_weights: bool = False) -> bool:
        return False

    def model_use_more_vram(
        self,
        extra_memory: int,
        force_patch_weights: bool = False,
    ) -> int:
        return self.partially_load(
            self.load_device, extra_memory,
            force_patch_weights=force_patch_weights,
        )

    # ─── Registration with comfy.model_management ────────────────────────
    def register(self):
        """Add a ``LoadedModel`` wrapping this patcher to
        ``comfy.model_management.current_loaded_models``. Idempotent.

        After registration, ComfyUI sees the pipeline in VRAM accounting
        and may evict it via ``free_memory``.

        IMPORTANT: we call ``loaded.model_load()`` after construction.
        ``LoadedModel.__init__`` leaves ``self.real_model = None`` (the
        attribute, not a weakref). ``real_model`` is only set up as a
        weakref by ``model_load()``. ComfyUI normally calls
        ``model_load()`` itself when a workflow consumes a model — but
        our patcher is registered out-of-band (no node references it
        directly as a model output) so ComfyUI never loads it. If we
        leave ``real_model = None``, the next
        ``cleanup_models_gc()``/``cleanup_models()`` call invokes
        ``is_dead()`` → ``self.real_model()`` → ``None()`` →
        ``TypeError: 'NoneType' object is not callable``, killing the
        prompt_worker thread. ``model_load()`` is safe here: our
        ``partially_load()`` short-circuits with ``return 0`` when the
        pipeline is already on ``load_device`` (which it is — callers do
        ``pipeline.cuda()`` before wrapping).
        """
        mm = _mm()
        for existing in mm.current_loaded_models:
            try:
                if existing.model is self:
                    return existing
            except Exception:
                continue
        loaded = mm.LoadedModel(self)
        # Set up real_model weakref + model_finalizer. Without this,
        # is_dead() crashes the prompt_worker (see docstring above).
        loaded.model_load()
        mm.current_loaded_models.append(loaded)
        log.info(
            "[PipelinePatcher] registered %s (%.2fGB, type=%s) with ComfyUI",
            self.name, self._size_bytes / 1e9, self.pipeline_type,
        )
        return loaded

    def unregister(self) -> None:
        """Remove from ``current_loaded_models`` and move pipeline to CPU."""
        mm = _mm()
        for i, existing in enumerate(list(mm.current_loaded_models)):
            try:
                if existing.model is self:
                    mm.current_loaded_models.pop(i)
                    try:
                        if existing.model_finalizer is not None:
                            existing.model_finalizer.detach()
                    except Exception:
                        pass
                    break
            except Exception:
                continue
        self.detach()
        log.info(
            "[PipelinePatcher] unregistered %s (type=%s)",
            self.name, self.pipeline_type,
        )

    # ─── Diagnostics ─────────────────────────────────────────────────────
    def __repr__(self) -> str:
        size_gb = self._size_bytes / 1e9
        dev = self.current_loaded_device()
        return (
            f"<PipelinePatcher {self.pipeline_type}/{self.name} "
            f"size={size_gb:.2f}GB dev={dev}>"
        )


# ─── Per-process cache helpers ─────────────────────────────────────────────
_pipeline_cache: dict[str, PipelinePatcher] = {}
_pipeline_cache_lock = threading.Lock()


def register_pipeline(
    key: str,
    patcher: PipelinePatcher,
) -> PipelinePatcher:
    """Cache + register a pipeline patcher. Thread-safe."""
    with _pipeline_cache_lock:
        existing = _pipeline_cache.get(key)
        if existing is not None and existing.model is not patcher.model:
            existing.unregister()
            _pipeline_cache[key] = patcher
        elif existing is not None:
            return existing
        else:
            _pipeline_cache[key] = patcher
    patcher.register()
    return patcher


def get_pipeline(key: str) -> Optional[PipelinePatcher]:
    """Look up a cached patcher by key. Returns None if not cached."""
    with _pipeline_cache_lock:
        return _pipeline_cache.get(key)


def unregister_pipeline(key: str) -> bool:
    """Unregister + evict a cached patcher. Returns True if found."""
    with _pipeline_cache_lock:
        patcher = _pipeline_cache.pop(key, None)
    if patcher is None:
        return False
    patcher.unregister()
    return True


def unregister_all_pipelines(prefix: str = "") -> int:
    """Unregister every cached patcher whose key starts with ``prefix``.

    Returns the count evicted. Pass prefix="trellis" to evict only
    trellis pipelines, or prefix="" (default) to evict ALL.
    """
    with _pipeline_cache_lock:
        keys_to_evict = [k for k in _pipeline_cache if k.startswith(prefix)]
        patchers = [_pipeline_cache.pop(k) for k in keys_to_evict]
    for p in patchers:
        p.unregister()
    if patchers:
        log.info(
            "[PipelinePatcher] evicted %d pipeline(s) matching prefix=%r",
            len(patchers), prefix,
        )
    return len(patchers)


def list_registered_pipelines() -> list[dict]:
    """Return a list of registered pipeline diagnostics. For health routes."""
    with _pipeline_cache_lock:
        out = []
        for key, p in _pipeline_cache.items():
            try:
                out.append({
                    "key": key,
                    "type": p.pipeline_type,
                    "name": p.name,
                    "size_gb": round(p._size_bytes / 1e9, 2),
                    "device": str(p.current_loaded_device()),
                })
            except Exception:
                continue
        return out


__all__ = [
    "PipelinePatcher",
    "register_pipeline",
    "get_pipeline",
    "unregister_pipeline",
    "unregister_all_pipelines",
    "list_registered_pipelines",
]
