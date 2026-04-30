# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    _WANDB_INSTALLED = True
except ModuleNotFoundError:
    _WANDB_INSTALLED = False


class WandbSummaryWriter(SummaryWriter):
    """Summary writer for Weights and Biases.

    All wandb calls are best-effort: any connection failure, process error, or
    API error only prints a one-time warning and training continues normally.
    TensorBoard logging is always active regardless of wandb status.
    """

    def __init__(self, log_dir: str, flush_secs: int, cfg):
        super().__init__(log_dir, flush_secs)

        self._wandb_ok = False  # set to True only when wandb.init() succeeds

        if not _WANDB_INSTALLED:
            print("[WandbWriter] wandb not installed — TensorBoard only.")
            return

        run_name = os.path.split(log_dir)[-1]

        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config.")

        entity = os.environ.get("WANDB_USERNAME", None)

        # Try init strategies in order of preference.
        # Each failure falls through to the next without crashing.
        # NOTE: "thread" must come before "fork" — fork after CUDA initialization
        # can deadlock the child process (CUDA contexts are not fork-safe).
        # This is especially likely when optimizer state tensors are loaded to GPU
        # before wandb.init() is called (e.g., is_full_resume=True in Stage 2).
        _init_strategies = [
            {"settings": wandb.Settings(start_method="thread")},
            {"settings": wandb.Settings(start_method="fork")},
            {"mode": "offline"},   # no network needed; syncs later with `wandb sync`
        ]

        for kwargs in _init_strategies:
            try:
                wandb.init(project=project, entity=entity, name=run_name, **kwargs)
                wandb.config.update({"log_dir": log_dir})
                self._wandb_ok = True
                mode = kwargs.get("mode", kwargs.get("settings", "default"))
                print(f"[WandbWriter] wandb initialized (strategy={mode}).")
                break
            except Exception as e:
                print(f"[WandbWriter] wandb.init attempt failed ({kwargs}): {e}")

        if not self._wandb_ok:
            print("[WandbWriter] All wandb init strategies failed — TensorBoard only.")

        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

    # ── wandb config ──────────────────────────────────────────────────────────

    def store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        if not self._wandb_ok:
            return
        try:
            wandb.config.update({"runner_cfg": runner_cfg})
            wandb.config.update({"policy_cfg": policy_cfg})
            wandb.config.update({"alg_cfg": alg_cfg})
            try:
                wandb.config.update({"env_cfg": env_cfg.to_dict()})
            except Exception:
                wandb.config.update({"env_cfg": asdict(env_cfg)})
        except Exception as e:
            print(f"[WandbWriter] store_config failed: {e}")

    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

    # ── scalar logging ────────────────────────────────────────────────────────

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        if not self._wandb_ok:
            return
        try:
            wandb.log({self._map_path(tag): scalar_value}, step=global_step)
        except Exception:
            pass  # silent: wandb disconnect should not interrupt training

    # ── file / model saving ───────────────────────────────────────────────────

    def save_model(self, model_path, iter):
        if not self._wandb_ok:
            return
        try:
            wandb.save(model_path, base_path=os.path.dirname(model_path))
        except Exception as e:
            print(f"[WandbWriter] save_model failed: {e}")

    def save_file(self, path, iter=None):
        if not self._wandb_ok:
            return
        try:
            wandb.save(path, base_path=os.path.dirname(path))
        except Exception as e:
            print(f"[WandbWriter] save_file failed: {e}")

    def stop(self):
        if not self._wandb_ok:
            return
        try:
            wandb.finish()
        except Exception:
            pass

    # ── internal ──────────────────────────────────────────────────────────────

    def _map_path(self, path):
        return self.name_map.get(path, path)
