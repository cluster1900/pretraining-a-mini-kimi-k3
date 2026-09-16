"""
Full-State Checkpoint Manager.
Ensures bit-exact recovery across process restarts by persisting:
- Model parameters
- Optimizer momentum & state
- DataLoader cursors
- All 4 RNG states (Python random, NumPy, PyTorch CPU, PyTorch CUDA)
- SpikeGuard EMA
- Telemetry summary
Uses atomic writes with a COMPLETE marker file to avoid loading partial checkpoints.
"""

import os
import shutil
import random
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist


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
        self.best_metric = None
        self.best_step = None

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
    ) -> Path:
        """
        Saves a complete checkpoint atomically.
        """
        ckpt_name = f"step_{step:06d}"
        target_dir = self.checkpoint_dir / ckpt_name
        tmp_dir = self.checkpoint_dir / f".tmp_{ckpt_name}"
        
        if rank == 0 and tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        if dist.is_available() and dist.is_initialized(): dist.barrier()
        tmp_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Model weights
        torch.save(model.state_dict(), tmp_dir / "model.pt")
        
        # 2. Rank-local optimizer state (each DDP replica owns its state)
        torch.save(optimizer.state_dict(), tmp_dir / f"optimizer_rank{rank}.pt")
        
        # 3. RNG States
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        torch.save(rng_state, tmp_dir / f"rng_rank{rank}.pt")
        
        # 4. Metadata & State
        meta = {
            "step": step,
            "spike_guard": spike_guard.state_dict(),
            "data_loader": data_loader_state if rank == 0 else None,
            "world_size": world_size,
        }
        if extra_meta:
            meta.update(extra_meta)
        if "val_loss" in meta and rank == 0:
            metric = float(meta["val_loss"])
            if self.best_metric is None or metric < self.best_metric:
                self.best_metric, self.best_step = metric, step
                meta["is_best"] = True
                meta["best_metric"] = metric
        torch.save(meta, tmp_dir / "meta.pt")
        
        # Each rank writes its own loader cursor. Marker is written only by
        # rank 0 after all ranks have finished writing their state.
        if data_loader_state is not None:
            torch.save(data_loader_state, tmp_dir / f"loader_rank{rank}.pt")
        if dist.is_available() and dist.is_initialized(): dist.barrier()
        if rank == 0:
            (tmp_dir / "COMPLETE").write_text(f"step={step}\nworld_size={world_size}\n")
        
        # Atomic rename
            if target_dir.exists(): shutil.rmtree(target_dir)
            tmp_dir.rename(target_dir)
            if meta.get("is_best"):
                best = self.checkpoint_dir / "best"
                best_tmp = self.checkpoint_dir / ".best.tmp"
                if best_tmp.exists(): shutil.rmtree(best_tmp)
                shutil.copytree(target_dir, best_tmp)
                if best.exists(): shutil.rmtree(best)
                best_tmp.rename(best)
        if dist.is_available() and dist.is_initialized(): dist.barrier()
        
        # Clean up old checkpoints based on retention policy
        self._prune_old_checkpoints(step)
        return target_dir

    def _prune_old_checkpoints(self, current_step: int) -> None:
        """
        Retains:
        - The last `keep_last_n` checkpoints
        - Checkpoints whose step is a multiple of `milestone_interval` (e.g. every 5000 steps)
        - Step 0
        """
        dirs = sorted([d for d in self.checkpoint_dir.glob("step_*") if d.is_dir() and (d / "COMPLETE").exists()])
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

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        spike_guard: Optional[Any] = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        Finds and loads the latest valid checkpoint containing COMPLETE.
        Restores model, optimizer, RNG, and SpikeGuard.
        Returns the metadata dictionary.
        """
        valid_dirs = sorted([d for d in self.checkpoint_dir.glob("step_*") if d.is_dir() and (d / "COMPLETE").exists()])
        if not valid_dirs:
            return None
            
        latest_dir = valid_dirs[-1]
        print(f"[CheckpointManager] Loading latest checkpoint from: {latest_dir}")
        
        # Load weights
        model.load_state_dict(torch.load(latest_dir / "model.pt", map_location="cpu"))
        
        # Load optimizer
        opt_file = latest_dir / f"optimizer_rank{rank}.pt"
        if optimizer is not None and opt_file.exists():
            optimizer.load_state_dict(torch.load(opt_file, map_location="cpu"))
            
        # Load RNG states
        rng_file = latest_dir / f"rng_rank{rank}.pt"
        if rng_file.exists():
            rng = torch.load(rng_file, map_location="cpu")
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch_cpu"])
            if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
                torch.cuda.set_rng_state(rng["torch_cuda"])
                
        # Load meta & spike guard
        meta = torch.load(latest_dir / "meta.pt", map_location="cpu")
        saved_world = int(meta.get("world_size", 1))
        if saved_world != world_size:
            raise ValueError(f"Checkpoint world_size={saved_world} but current world_size={world_size}; refusing unsafe resume")
        loader_file = latest_dir / f"loader_rank{rank}.pt"
        if loader_file.exists():
            meta["data_loader"] = torch.load(loader_file, map_location="cpu")
        if spike_guard is not None and "spike_guard" in meta:
            spike_guard.load_state_dict(meta["spike_guard"])
            
        return meta
