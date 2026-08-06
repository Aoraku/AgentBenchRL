"""Content-addressed YAML population manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

import yaml


AgentKind = Literal[
    "train_human", "test_human", "baseline", "checkpoint", "champion"
]
AgentProtocol = Literal["line_json", "snakego_official"]
_KINDS = {"train_human", "test_human", "baseline", "checkpoint", "champion"}
_PROTOCOLS = {"line_json", "snakego_official"}
_ROLES = {"player", "player_0", "player_1"}
_TRAINING_KINDS = _KINDS - {"test_human"}


@dataclass(frozen=True, slots=True)
class PopulationEntry:
    """One immutable, content-addressed launch specification."""

    agent_id: str
    kind: AgentKind
    content_hash: str
    population_root: Path
    command: tuple[str, ...]
    protocol: AgentProtocol
    roles: tuple[str, ...]
    resource_limits: Mapping[str, Any]
    provenance: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PopulationManifest:
    """A frozen population whose launch paths stay inside one declared root."""

    population_root: Path
    protocol_version: str
    entries: tuple[PopulationEntry, ...]
    content_hash: str

    @classmethod
    def from_yaml(cls, path: str | Path) -> PopulationManifest:
        manifest_path = Path(path)
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("population manifest must be a mapping")
        root_value = raw.get("population_root")
        if not isinstance(root_value, str) or not root_value:
            raise ValueError("population_root must be a non-empty relative path")
        relative_root = Path(root_value)
        if relative_root.is_absolute():
            raise ValueError("population_root must be relative to the manifest")
        root = _within_root(manifest_path.parent.resolve(), relative_root)
        return cls.from_data(raw, population_root=root)

    @classmethod
    def from_data(
        cls, data: Mapping[str, Any], *, population_root: str | Path
    ) -> PopulationManifest:
        root = Path(population_root).resolve()
        raw_entries = data.get("agents")
        if not isinstance(raw_entries, list):
            raise ValueError("agents must be a list")
        entries: list[PopulationEntry] = []
        seen: set[str] = set()
        for raw in raw_entries:
            if not isinstance(raw, Mapping):
                raise ValueError("each agent must be a mapping")
            agent_id = _nonempty_string(raw.get("agent_id"), "agent_id")
            if agent_id in seen:
                raise ValueError(f"duplicate agent_id: {agent_id}")
            seen.add(agent_id)
            kind = _nonempty_string(raw.get("kind"), "kind")
            if kind not in _KINDS:
                raise ValueError(f"unsupported population kind: {kind}")
            command_raw = raw.get("command")
            if not isinstance(command_raw, list) or not command_raw:
                raise ValueError("command must be a non-empty list")
            command = tuple(
                _nonempty_string(part, "command element") for part in command_raw
            )
            protocol = _nonempty_string(raw.get("protocol", "line_json"), "protocol")
            if protocol not in _PROTOCOLS:
                raise ValueError(f"unsupported agent protocol: {protocol}")
            executable = _within_root(root, Path(command[0]))
            expected_hash = _nonempty_string(raw.get("content_hash"), "content_hash")
            actual_hash = _file_hash(executable)
            if expected_hash != actual_hash:
                raise ValueError(
                    f"content hash mismatch for {agent_id}: expected {expected_hash}, "
                    f"found {actual_hash}"
                )
            roles_raw = raw.get("roles", [])
            if not isinstance(roles_raw, list):
                raise ValueError("roles must be a list")
            roles = tuple(_nonempty_string(role, "role") for role in roles_raw)
            unsupported_roles = set(roles) - _ROLES
            if unsupported_roles:
                raise ValueError(
                    f"unsupported population role: {sorted(unsupported_roles)[0]}"
                )
            limits = raw.get("resource_limits", {})
            provenance = raw.get("provenance", {})
            if not isinstance(limits, Mapping) or not isinstance(provenance, Mapping):
                raise ValueError("resource_limits and provenance must be mappings")
            entries.append(
                PopulationEntry(
                    agent_id=agent_id,
                    kind=kind,  # type: ignore[arg-type]
                    content_hash=expected_hash,
                    population_root=root,
                    command=command,
                    protocol=protocol,  # type: ignore[arg-type]
                    roles=roles,
                    resource_limits=_freeze_mapping(limits),
                    provenance=_freeze_mapping(provenance),
                )
            )
        canonical = json.dumps(
            data, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        return cls(
            population_root=root,
            protocol_version=str(data.get("protocol_version", "1")),
            entries=tuple(entries),
            content_hash=f"sha256:{hashlib.sha256(canonical).hexdigest()}",
        )

    def entry(self, agent_id: str) -> PopulationEntry:
        for entry in self.entries:
            if entry.agent_id == agent_id:
                return entry
        raise KeyError(agent_id)

    def training_entries(self) -> tuple[PopulationEntry, ...]:
        """Return only populations allowed to affect training decisions."""
        return tuple(entry for entry in self.entries if entry.kind in _TRAINING_KINDS)


def _within_root(root: Path, relative: Path) -> Path:
    if relative.is_absolute():
        raise ValueError("command paths must be relative to the population root")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("command path escapes the population root")
    return resolved


def _file_hash(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"command executable is not readable: {path.name}") from exc
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value
