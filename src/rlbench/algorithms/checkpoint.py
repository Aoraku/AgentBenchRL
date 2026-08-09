"""Atomic, schema-versioned policy checkpoints with complete RNG state."""

from __future__ import annotations

import copy
import os
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PolicyCheckpoint:
    """A validated checkpoint payload that can be restored atomically."""

    path: Path
    _payload: Mapping[str, Any] = field(repr=False)

    @property
    def schema_version(self) -> int:
        return int(self._payload["schema_version"])

    @classmethod
    def save(
        cls,
        path: str | Path,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        replay_state: Mapping[str, Any] | None = None,
        trainer_state: Mapping[str, Any] | None = None,
        custom_rng_state: Mapping[str, Any] | None = None,
    ) -> PolicyCheckpoint:
        destination = Path(path)
        payload: dict[str, Any] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "replay_state": dict(replay_state or {}),
            "trainer_state": dict(trainer_state or {}),
            "rng_state": {
                "python": random.getstate(),
                "numpy_global": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
                "custom": dict(custom_rng_state or {}),
            },
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        try:
            torch.save(payload, temporary_name)
            with open(temporary_name, "rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return cls(destination, payload)

    @classmethod
    def load(
        cls, path: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> PolicyCheckpoint:
        source = Path(path)
        try:
            payload = torch.load(source, map_location=map_location, weights_only=False)
        except Exception as exc:
            raise ValueError(f"invalid policy checkpoint: {source}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("policy checkpoint must contain a mapping")
        required = {"schema_version", "model_state", "rng_state", "trainer_state"}
        if not required.issubset(payload):
            raise ValueError("policy checkpoint is missing required fields")
        if int(payload["schema_version"]) != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported policy checkpoint schema version")
        if not isinstance(payload["model_state"], Mapping):
            raise ValueError("invalid checkpoint model state")
        if not isinstance(payload["rng_state"], Mapping):
            raise ValueError("invalid checkpoint RNG state")
        return cls(source, payload)

    @property
    def trainer_state(self) -> Mapping[str, Any]:
        value = self._payload["trainer_state"]
        if not isinstance(value, Mapping):
            raise ValueError("invalid checkpoint trainer state")
        return value

    @property
    def model_state(self) -> Mapping[str, Any]:
        return self._payload["model_state"]

    @property
    def replay_state(self) -> Mapping[str, Any]:
        value = self._payload.get("replay_state", {})
        if not isinstance(value, Mapping):
            raise ValueError("invalid checkpoint replay state")
        return value

    @property
    def custom_rng_state(self) -> Mapping[str, Any]:
        value = self._payload["rng_state"].get("custom", {})
        if not isinstance(value, Mapping):
            raise ValueError("invalid checkpoint custom RNG state")
        return value

    def restore(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
    ) -> None:
        self.validate_restore(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        model_snapshot = {
            name: tensor.detach().clone() for name, tensor in model.state_dict().items()
        }
        optimizer_snapshot = (
            copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
        )
        scheduler_snapshot = (
            copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None
        )
        rng_snapshot = _capture_rng_state()
        try:
            self._apply_restore(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
        except Exception as exc:
            model.load_state_dict(model_snapshot, strict=True)
            if optimizer is not None and optimizer_snapshot is not None:
                optimizer.load_state_dict(optimizer_snapshot)
            if scheduler is not None and scheduler_snapshot is not None:
                scheduler.load_state_dict(scheduler_snapshot)
            _apply_rng_state(rng_snapshot)
            raise ValueError("policy checkpoint restore failed transactionally") from exc

    def validate_restore(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
    ) -> None:
        """Stage every generic payload component without mutating live objects."""
        model_state = self._payload["model_state"]
        current_state = model.state_dict()
        if set(model_state) != set(current_state):
            raise ValueError("checkpoint model keys do not match")
        for name, tensor in model_state.items():
            current = current_state[name]
            if not isinstance(tensor, torch.Tensor) or tensor.shape != current.shape:
                raise ValueError(f"checkpoint tensor does not match model: {name}")
        optimizer_state = self._payload.get("optimizer_state")
        if optimizer is not None and not isinstance(optimizer_state, Mapping):
            raise ValueError("checkpoint does not contain optimizer state")
        scheduler_state = self._payload.get("scheduler_state")
        if scheduler is not None and not isinstance(scheduler_state, Mapping):
            raise ValueError("checkpoint does not contain scheduler state")

        try:
            staged_model = copy.deepcopy(model)
            staged_model.load_state_dict(model_state, strict=True)
            if optimizer is not None:
                staged_optimizer = copy.deepcopy(optimizer)
                staged_optimizer.load_state_dict(optimizer_state)
                _validate_optimizer_parameter_states(staged_optimizer)
            if scheduler is not None:
                staged_scheduler = copy.deepcopy(scheduler)
                staged_scheduler.load_state_dict(scheduler_state)
            _validate_rng_state(self._payload["rng_state"])
        except Exception as exc:
            raise ValueError(
                "invalid policy checkpoint optimizer or restore state"
            ) from exc

    def _apply_restore(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
    ) -> None:
        model_state = self._payload["model_state"]
        optimizer_state = self._payload.get("optimizer_state")
        scheduler_state = self._payload.get("scheduler_state")
        model.load_state_dict(model_state, strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(optimizer_state)
        if scheduler is not None:
            scheduler.load_state_dict(scheduler_state)
        _apply_rng_state(self._payload["rng_state"])


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None,
    }


def _validate_optimizer_parameter_states(
    optimizer: torch.optim.Optimizer,
) -> None:
    """Reject malformed full-parameter Adam-family tensor slots."""
    full_parameter_slots = {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            for slot_name, slot in optimizer.state.get(parameter, {}).items():
                if slot_name not in full_parameter_slots:
                    continue
                if not isinstance(slot, torch.Tensor) or slot.ndim == 0:
                    continue
                if slot.shape != parameter.shape:
                    raise ValueError(
                        "optimizer tensor slot shape does not match parameter: "
                        f"{slot_name}"
                    )


def _validate_rng_state(state: Any) -> None:
    if not isinstance(state, Mapping):
        raise ValueError("invalid checkpoint RNG state")
    required = {"python", "numpy_global", "torch_cpu", "torch_cuda", "custom"}
    if not required.issubset(state):
        raise ValueError("checkpoint RNG state is incomplete")
    python_rng = random.Random()
    python_rng.setstate(state["python"])
    numpy_rng = np.random.RandomState()
    numpy_rng.set_state(state["numpy_global"])
    torch_rng = torch.Generator(device="cpu")
    torch_state = state["torch_cpu"]
    if not isinstance(torch_state, torch.Tensor):
        raise ValueError("invalid Torch RNG state")
    torch_rng.set_state(torch_state.cpu())
    cuda_states = state["torch_cuda"]
    if cuda_states is not None and not (
        isinstance(cuda_states, list)
        and all(isinstance(value, torch.Tensor) for value in cuda_states)
    ):
        raise ValueError("invalid CUDA RNG state")
    if not isinstance(state["custom"], Mapping):
        raise ValueError("invalid custom RNG state")


def _apply_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(
            [cuda_state.cpu() for cuda_state in state["torch_cuda"]]
        )
