"""
Full-State Checkpoint Manager.
Ensures bit-exact recovery across process restarts by persisting:
- Model parameters (rank 0, once)
- Optimizer momentum & state and GradScaler state (rank 0, once: under DDP
  every rank holds identical gradients after the all-reduce, so the
  replicated Muon/AdamW and scaler states are identical across ranks)
- DataLoader cursors (per rank: every rank reads its own token slices)
- All 4 RNG states (Python random, NumPy, PyTorch CPU, PyTorch CUDA; per rank)
- SpikeGuard EMA, best validation metric
- Telemetry summary

Durability: every file is fsynced, then the temporary directory, then the
``COMPLETE`` marker, and the parent directory after the atomic rename. Only
directories with ``COMPLETE`` are considered. If the newest one cannot be read
(e.g. a truncated file after a crash), ``load_latest`` falls back to the
previous complete checkpoint with a loud warning; all ranks agree on which
checkpoint they load.

Layout: ``step_{step:06d}/{model.pt, optimizer.pt, scaler.pt, meta.pt,
rng_rank{r}.pt, loader_rank{r}.pt, COMPLETE}``. Checkpoints written by the
older layout (``optimizer_rank{r}.pt``/``scaler_rank{r}.pt``) still load.
"""

import os
import shutil
import random
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist


class CheckpointSafetyError(ValueError):
    """The checkpoint belongs to a different run; never fall back silently."""


def _dist_on() -> bool:
    return dist.is_available() and dist.is_initialized()


def _barrier() -> None:
    if _dist_on():
        dist.barrier()


def _fsync_path(path: Path) -> None:
    """fsync a file or a directory (directory fsync is a no-op where unsupported)."""
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY") and path.is_dir():
        flags |= os.O_DIRECTORY
    try:
        fd = os.open(str(path), flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _save(obj: Any, path: Path) -> None:
    with open(path, "wb") as handle:
        torch.save(obj, handle)
        handle.flush()
        os.fsync(handle.fileno())


def _write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _all_ranks_ok(ok: bool) -> bool:
    """MIN-reduce a success flag so every rank takes the same branch."""
    if not _dist_on():
        return ok
    device = torch.device("cpu")
    if dist.get_backend() == "nccl":
        device = torch.device("cuda", torch.cuda.current_device())
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _link_or_copy_tree(src: Path, dst: Path) -> None:
    """Hard-link every file of ``src`` into ``dst`` (copy if links are unsupported)."""
    dst.mkdir(parents=True, exist_ok=False)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            _link_or_copy_tree(item, target)
            continue
        try:
            os.link(item, target)
        except OSError:
            shutil.copy2(item, target)
            _fsync_path(target)
    _fsync_path(dst)


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        keep_last_n: int = 3,
        save_interval: int = 1000,
        milestone_interval: int = 5000,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.save_interval = save_interval
        self.milestone_interval = milestone_interval
        self.best_metric: Optional[float] = None
        self.best_step: Optional[int] = None

    # ------------------------------------------------------------------ save
    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        spike_guard: Any,
        data_loader_state: Optional[Dict[str, Any]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
        rank: int = 0,
        world_size: int = 1,
        scaler: Optional[Any] = None,
    ) -> Path:
        """
        Saves a complete checkpoint atomically.

        ``extra_meta["val_loss"]`` (validation backbone LM loss), when present,
        drives best-checkpoint selection (lower is better).
        """
        ckpt_name = f"step_{step:06d}"
        target_dir = self.checkpoint_dir / ckpt_name
        tmp_dir = self.checkpoint_dir / f".tmp_{ckpt_name}"

        if rank == 0 and tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        _barrier()
        tmp_dir.mkdir(parents=True, exist_ok=True)

        meta: Dict[str, Any] = {
            "step": step,
            "spike_guard": spike_guard.state_dict(),
            "world_size": world_size,
            "layout": 2,
        }
        if extra_meta:
            meta.update(extra_meta)
        # Every rank evaluates the same all-reduced metric, so best tracking
        # stays identical across ranks without communication.
        is_best = False
        if meta.get("val_loss") is not None:
            metric = float(meta["val_loss"])
            if self.best_metric is None or metric < self.best_metric:
                self.best_metric, self.best_step = metric, step
                is_best = True
        meta["is_best"] = is_best
        meta["best_metric"] = self.best_metric
        meta["best_step"] = self.best_step

        # Replicated state is written once; rank-local state by every rank.
        if rank == 0:
            _save(model.state_dict(), tmp_dir / "model.pt")
            _save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
            if scaler is not None:
                _save(scaler.state_dict(), tmp_dir / "scaler.pt")
            _save(meta, tmp_dir / "meta.pt")
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        _save(rng_state, tmp_dir / f"rng_rank{rank}.pt")
        if data_loader_state is not None:
            _save(data_loader_state, tmp_dir / f"loader_rank{rank}.pt")
        _barrier()
        if rank == 0:
            missing = [
                name for name in [f"rng_rank{r}.pt" for r in range(world_size)]
                if not (tmp_dir / name).is_file()
            ]
            if data_loader_state is not None:
                missing += [f"loader_rank{r}.pt" for r in range(world_size)
                            if not (tmp_dir / f"loader_rank{r}.pt").is_file()]
            if missing:
                raise RuntimeError(f"Checkpoint {tmp_dir} is missing rank files: {missing}")
            _fsync_path(tmp_dir)
            _write_text(tmp_dir / "COMPLETE", f"step={step}\nworld_size={world_size}\n")
            _fsync_path(tmp_dir)
            old_dir = None
            if target_dir.exists():
                old_dir = self.checkpoint_dir / f".old_{ckpt_name}"
                if old_dir.exists():
                    shutil.rmtree(old_dir)
                target_dir.rename(old_dir)
            tmp_dir.rename(target_dir)
            _fsync_path(self.checkpoint_dir)
            if old_dir is not None:
                shutil.rmtree(old_dir, ignore_errors=True)
            if is_best:
                self._publish_best(target_dir)
            self._prune_old_checkpoints(step)
        _barrier()
        return target_dir

    def _publish_best(self, target_dir: Path) -> None:
        best = self.checkpoint_dir / "best"
        best_tmp = self.checkpoint_dir / ".best.tmp"
        best_old = self.checkpoint_dir / ".best.old"
        for stale in (best_tmp, best_old):
            if stale.exists():
                shutil.rmtree(stale)
        # Hard links: no extra disk and no multi-GB copy; checkpoint files are
        # never modified in place, and pruning step_* only drops one link.
        _link_or_copy_tree(target_dir, best_tmp)
        if best.exists():
            best.rename(best_old)
        best_tmp.rename(best)
        _fsync_path(self.checkpoint_dir)
        if best_old.exists():
            shutil.rmtree(best_old, ignore_errors=True)

    def _complete_dirs(self) -> List[Path]:
        return sorted(
            d for d in self.checkpoint_dir.glob("step_*") if d.is_dir() and (d / "COMPLETE").exists()
        )

    def _prune_old_checkpoints(self, current_step: int) -> None:
        """
        Retains:
        - The last `keep_last_n` checkpoints
        - Checkpoints whose step is a multiple of `milestone_interval` (e.g. every 5000 steps)
        - Step 0
        """
        dirs = self._complete_dirs()
        if len(dirs) <= self.keep_last_n:
            return

        recent_dirs = set(dirs[-self.keep_last_n:])
        for d in dirs:
            try:
                s = int(d.name.split("_")[1])
            except (IndexError, ValueError):
                continue

            # Keep milestones and the latest n
            if s == 0 or s % self.milestone_interval == 0 or d in recent_dirs:
                continue

            # Otherwise delete
            shutil.rmtree(d, ignore_errors=True)

    # ------------------------------------------------------------------ load
    @staticmethod
    def _first_existing(directory: Path, names: List[str]) -> Optional[Path]:
        for name in names:
            if (directory / name).is_file():
                return directory / name
        return None

    def _load_dir(
        self,
        ckpt_dir: Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        spike_guard: Optional[Any],
        rank: int,
        world_size: int,
        scaler: Optional[Any],
        expected_run_signature: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        meta = _load(ckpt_dir / "meta.pt")
        saved_world = int(meta.get("world_size", 1))
        if saved_world != world_size:
            raise CheckpointSafetyError(
                f"Checkpoint world_size={saved_world} but current world_size={world_size}; refusing unsafe resume"
            )
        if expected_run_signature is not None and meta.get("run_signature") != expected_run_signature:
            saved = meta.get("run_signature") or {}
            diff = sorted(
                k for k in set(saved) | set(expected_run_signature)
                if saved.get(k) != expected_run_signature.get(k)
            )
            raise CheckpointSafetyError(
                "Checkpoint run signature differs from current manifest/config; refusing unsafe resume "
                f"(differing fields: {diff})"
            )
        # Validate presence of every file before mutating anything.
        opt_file = self._first_existing(ckpt_dir, ["optimizer.pt", f"optimizer_rank{rank}.pt"])
        scaler_file = self._first_existing(ckpt_dir, ["scaler.pt", f"scaler_rank{rank}.pt"])
        rng_file = ckpt_dir / f"rng_rank{rank}.pt"
        model_file = ckpt_dir / "model.pt"
        if not model_file.is_file():
            raise FileNotFoundError(f"Missing model weights: {model_file}")
        if optimizer is not None and opt_file is None:
            raise FileNotFoundError(f"Missing optimizer state in {ckpt_dir}")
        if not rng_file.exists():
            raise FileNotFoundError(f"Missing RNG state: {rng_file}")
        if scaler is not None and scaler_file is None:
            raise FileNotFoundError(f"Missing GradScaler state in {ckpt_dir}")

        # Deserialize everything rank-local first so a corrupt file is detected
        # before the large model/optimizer loads mutate state.
        rng = _load(rng_file)
        loader_file = ckpt_dir / f"loader_rank{rank}.pt"
        loader_state = _load(loader_file) if loader_file.exists() else None
        scaler_state = _load(scaler_file) if scaler is not None else None

        state = _load(model_file)
        model.load_state_dict(state)
        del state
        if optimizer is not None:
            opt_state = _load(opt_file)
            optimizer.load_state_dict(opt_state)
            del opt_state
        if scaler is not None:
            scaler.load_state_dict(scaler_state)

        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"])
        if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
            torch.cuda.set_rng_state(rng["torch_cuda"])

        meta["data_loader"] = loader_state
        if spike_guard is not None and "spike_guard" in meta:
            spike_guard.load_state_dict(meta["spike_guard"])

        best_metric = meta.get("best_metric")
        best_step = meta.get("best_step")
        if best_metric is None:
            # Older checkpoints only marked the best directory itself.
            best_meta_file = self.checkpoint_dir / "best" / "meta.pt"
            if best_meta_file.is_file():
                try:
                    best_meta = _load(best_meta_file)
                    if best_meta.get("val_loss") is not None and int(best_meta.get("step", 1 << 62)) <= int(meta["step"]):
                        best_metric = float(best_meta["val_loss"])
                        best_step = int(best_meta["step"])
                except Exception as exc:  # pragma: no cover - best/ is advisory
                    warnings.warn(f"Could not read {best_meta_file}: {exc}")
        self.best_metric = None if best_metric is None else float(best_metric)
        self.best_step = None if best_step is None else int(best_step)
        return meta

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        spike_guard: Optional[Any] = None,
        rank: int = 0,
        world_size: int = 1,
        scaler: Optional[Any] = None,
        expected_run_signature: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Finds and loads the newest readable checkpoint containing COMPLETE.
        Restores model, optimizer, scaler, RNG, SpikeGuard and best metric.
        Returns the metadata dictionary (``meta["data_loader"]`` is this
        rank's loader state), or None if there is no complete checkpoint.

        A run-signature or world-size mismatch raises immediately: falling back
        would silently resume an older, equally foreign checkpoint. Any other
        read failure falls back to the previous complete checkpoint.
        """
        candidates = self._complete_dirs()
        if not candidates:
            return None
        failures = []
        for ckpt_dir in reversed(candidates):
            if rank == 0:
                print(f"[CheckpointManager] Loading checkpoint from: {ckpt_dir}")
            error: Optional[BaseException] = None
            meta = None
            try:
                meta = self._load_dir(ckpt_dir, model, optimizer, spike_guard, rank, world_size,
                                      scaler, expected_run_signature)
            except CheckpointSafetyError:
                raise
            except Exception as exc:  # corrupt / truncated / missing files
                error = exc
            if _all_ranks_ok(error is None):
                if failures and rank == 0:
                    print("!" * 80)
                    print(f"[CheckpointManager] WARNING: resumed from OLDER checkpoint {ckpt_dir} because "
                          f"newer checkpoint(s) failed to load: {failures}")
                    print("!" * 80)
                return meta
            reason = f"{type(error).__name__}: {error}" if error is not None else "failed on another rank"
            failures.append((ckpt_dir.name, reason))
            msg = (f"[CheckpointManager] WARNING: rank {rank} could not use checkpoint {ckpt_dir} "
                   f"({reason}); falling back to the previous COMPLETE checkpoint")
            print("!" * 80 + "\n" + msg + "\n" + "!" * 80)
            warnings.warn(msg)
        raise RuntimeError(f"No complete checkpoint in {self.checkpoint_dir} could be loaded: {failures}")
