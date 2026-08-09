"""Legal-mask PUCT search with actor-aware zero-sum value backup."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
from numpy.typing import NDArray

from rlbench.game import DiscreteGame, Observation, clone_game

from .config import AlphaZeroConfig
from .network import BatchEvaluator


@dataclass(slots=True)
class _Node:
    to_play: int | None
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    virtual_visit_count: int = 0
    children: dict[int, _Node] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def select_child(self, c_puct: float) -> tuple[int, _Node]:
        if self.to_play is None:
            raise RuntimeError("cannot select from a node with an unknown actor")
        total_visits = sum(
            child.visit_count + child.virtual_visit_count
            for child in self.children.values()
        )
        scale = np.sqrt(max(1, total_visits))

        def score(item: tuple[int, _Node]) -> tuple[float, int]:
            action, child = item
            visits = child.visit_count + child.virtual_visit_count
            if visits:
                if child.to_play is None:
                    raise RuntimeError("a visited child must have a resolved actor")
                real_value_sum = _convert_value(
                    child.value_sum,
                    from_player=child.to_play,
                    to_player=self.to_play,
                )
                parent_value = (
                    real_value_sum - child.virtual_visit_count
                ) / visits
            else:
                parent_value = 0.0
            exploration = c_puct * child.prior * scale / (1 + visits)
            return parent_value + exploration, -action

        return max(self.children.items(), key=score)


@dataclass(slots=True)
class _PendingLeaf:
    node: _Node
    path: list[_Node]
    observation: Observation | None
    legal_mask: NDArray[np.bool_] | None
    terminal_value: float | None


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Immutable root statistics exported as the deployed policy."""

    action: int
    visit_counts: NDArray[np.int64]
    visit_policy: NDArray[np.float32]
    action_values: NDArray[np.float32]
    root_priors: NDArray[np.float32]
    expanded_actions: tuple[int, ...]
    completed_simulations: int


class MCTS:
    """PUCT search that clones every simulation from the caller-owned root."""

    def __init__(
        self,
        config: AlphaZeroConfig,
        evaluator: BatchEvaluator,
        *,
        seed: int = 0,
    ) -> None:
        self.config = config
        self.evaluator = evaluator
        self.rng = np.random.default_rng(seed)

    def search(
        self,
        game: DiscreteGame,
        *,
        training: bool = False,
        move_number: int = 0,
        deadline: float | None = None,
    ) -> SearchResult:
        if game.outcome(game.current_player()) is not None:
            raise ValueError("cannot search a terminal game")
        root = _Node(to_play=game.current_player())
        root_simulation = clone_game(game)
        root_mask = _search_action_mask(root_simulation)
        root_observation = root_simulation.observe(root.to_play)
        root_logits, root_values = self.evaluator.evaluate_batch(
            [root_observation], [root_mask]
        )
        self._expand(root, root_mask, root_logits[0])
        if training:
            self._add_root_noise(root)
        self._backup([root], float(root_values[0]))

        completed = 1
        while completed < self.config.simulations:
            if deadline is not None and time.monotonic() >= deadline:
                break
            batch_size = min(
                self.config.inference_batch_size,
                self.config.simulations - completed,
            )
            pending = [self._select_leaf(game, root) for _ in range(batch_size)]
            evaluable = [leaf for leaf in pending if leaf.terminal_value is None]
            logits_by_leaf: dict[int, NDArray[np.floating]] = {}
            values_by_leaf: dict[int, float] = {}
            if evaluable:
                observations = [
                    leaf.observation for leaf in evaluable if leaf.observation is not None
                ]
                masks = [
                    leaf.legal_mask for leaf in evaluable if leaf.legal_mask is not None
                ]
                logits, values = self.evaluator.evaluate_batch(observations, masks)
                if len(logits) != len(evaluable) or len(values) != len(evaluable):
                    raise ValueError("evaluator outputs must match the requested batch")
                for leaf, leaf_logits, leaf_value in zip(
                    evaluable, logits, values, strict=True
                ):
                    logits_by_leaf[id(leaf)] = leaf_logits
                    values_by_leaf[id(leaf)] = float(leaf_value)

            for leaf in pending:
                self._release_virtual_loss(leaf.path)
                if leaf.terminal_value is None:
                    if leaf.legal_mask is None:
                        raise RuntimeError("non-terminal leaf is missing a legal mask")
                    if not leaf.node.children:
                        self._expand(
                            leaf.node,
                            leaf.legal_mask,
                            logits_by_leaf[id(leaf)],
                        )
                    leaf_value = values_by_leaf[id(leaf)]
                else:
                    leaf_value = leaf.terminal_value
                self._backup(leaf.path, leaf_value)
            completed += len(pending)

        action_count = len(game.spec.action_names)
        counts = np.zeros(action_count, dtype=np.int64)
        values = np.zeros(action_count, dtype=np.float32)
        priors = np.zeros(action_count, dtype=np.float32)
        for action, child in root.children.items():
            counts[action] = child.visit_count
            values[action] = (
                _convert_value(
                    child.value,
                    from_player=child.to_play,
                    to_player=root.to_play,
                )
                if child.to_play is not None and root.to_play is not None
                else 0.0
            )
            priors[action] = child.prior
        total = int(counts.sum())
        if total:
            policy = (counts / total).astype(np.float32)
        else:
            policy = priors / priors.sum()
        action = self._select_action(
            counts=counts,
            policy=policy,
            training=training,
            move_number=move_number,
        )
        return SearchResult(
            action=action,
            visit_counts=counts,
            visit_policy=policy,
            action_values=values,
            root_priors=priors,
            expanded_actions=tuple(sorted(root.children)),
            completed_simulations=completed,
        )

    def _select_leaf(self, game: DiscreteGame, root: _Node) -> _PendingLeaf:
        simulation = clone_game(game)
        node = root
        path = [root]
        while node.children:
            if node.to_play != simulation.current_player():
                raise RuntimeError("MCTS node actor disagrees with game state")
            action, child = node.select_child(self.config.c_puct)
            record = simulation.step(action)
            next_player = simulation.current_player()
            if child.to_play is None:
                child.to_play = next_player
            elif child.to_play != next_player:
                raise RuntimeError("MCTS child actor is not deterministic")
            node = child
            path.append(node)
            if record.terminated:
                if node.to_play is None:
                    raise RuntimeError("terminal child actor was not resolved")
                outcome = simulation.outcome(node.to_play)
                if outcome is None:
                    raise ValueError("terminal transition must expose an outcome")
                self._apply_virtual_loss(path)
                return _PendingLeaf(
                    node=node,
                    path=path,
                    observation=None,
                    legal_mask=None,
                    terminal_value=float(outcome),
                )
        legal_mask = _search_action_mask(simulation)
        if node.to_play is None:
            raise RuntimeError("leaf actor was not resolved")
        observation = simulation.observe(node.to_play)
        self._apply_virtual_loss(path)
        return _PendingLeaf(
            node=node,
            path=path,
            observation=observation,
            legal_mask=legal_mask,
            terminal_value=None,
        )

    @staticmethod
    def _apply_virtual_loss(path: list[_Node]) -> None:
        for node in path:
            node.virtual_visit_count += 1

    @staticmethod
    def _release_virtual_loss(path: list[_Node]) -> None:
        for node in path:
            node.virtual_visit_count -= 1

    @staticmethod
    def _backup(path: list[_Node], leaf_value: float) -> None:
        for index in range(len(path) - 1, -1, -1):
            visited = path[index]
            if visited.to_play is None:
                raise RuntimeError("cannot back up through an unresolved actor")
            visited.visit_count += 1
            visited.value_sum += leaf_value
            if index:
                parent = path[index - 1]
                if parent.to_play is None:
                    raise RuntimeError("cannot back up to an unresolved actor")
                leaf_value = _convert_value(
                    leaf_value,
                    from_player=visited.to_play,
                    to_player=parent.to_play,
                )

    @staticmethod
    def _expand(
        node: _Node,
        legal_mask: NDArray[np.bool_],
        logits: NDArray[np.floating],
    ) -> None:
        priors = _masked_softmax(logits, legal_mask)
        node.children = {
            int(action): _Node(
                to_play=None,
                prior=float(priors[action]),
            )
            for action in np.flatnonzero(legal_mask)
        }

    def _add_root_noise(self, root: _Node) -> None:
        actions = tuple(sorted(root.children))
        if not actions or self.config.root_dirichlet_fraction == 0.0:
            return
        noise = self.rng.dirichlet(
            np.full(len(actions), self.config.root_dirichlet_alpha)
        )
        fraction = self.config.root_dirichlet_fraction
        for action, sample in zip(actions, noise, strict=True):
            child = root.children[action]
            child.prior = (1.0 - fraction) * child.prior + fraction * float(sample)

    def _select_action(
        self,
        *,
        counts: NDArray[np.int64],
        policy: NDArray[np.float32],
        training: bool,
        move_number: int,
    ) -> int:
        if not training or move_number >= self.config.temperature_moves:
            return int(np.argmax(counts if counts.sum() else policy))
        temperature = self.config.self_play_temperature
        if temperature <= 1e-8:
            return int(np.argmax(counts if counts.sum() else policy))
        if counts.sum() == 0:
            probabilities = policy.astype(np.float64)
        else:
            probabilities = counts.astype(np.float64) ** (1.0 / temperature)
            probabilities /= probabilities.sum()
        probability_sum = float(probabilities.sum())
        if (
            not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
            or not np.isfinite(probability_sum)
            or probability_sum <= 0.0
        ):
            raise ValueError("training action probabilities must be finite and non-negative")
        probabilities /= probability_sum
        return int(self.rng.choice(len(probabilities), p=probabilities))


def _search_action_mask(game: DiscreteGame) -> NDArray[np.bool_]:
    legal = np.asarray(game.legal_action_mask(), dtype=np.bool_)
    search_action_mask = getattr(game, "search_action_mask", None)
    if not callable(search_action_mask):
        return legal
    mask = np.asarray(search_action_mask(game.current_player()), dtype=np.bool_)
    if mask.shape != legal.shape or np.any(mask & ~legal) or not np.any(mask):
        raise ValueError("search action mask must be a non-empty legal subset")
    return mask


def _masked_softmax(
    logits: NDArray[np.floating], legal_mask: NDArray[np.bool_]
) -> NDArray[np.float32]:
    if logits.shape != legal_mask.shape:
        raise ValueError("evaluator logits must match the legal-action mask")
    if not legal_mask.any():
        raise ValueError("a non-terminal search node must have a legal action")
    result = np.zeros(logits.shape, dtype=np.float32)
    legal_logits = np.asarray(logits[legal_mask], dtype=np.float64)
    weights = np.exp(legal_logits - np.max(legal_logits))
    result[legal_mask] = (weights / weights.sum()).astype(np.float32)
    return result


def _convert_value(value: float, *, from_player: int, to_player: int) -> float:
    """Convert a zero-sum value between the actors attached to adjacent nodes."""
    return value if from_player == to_player else -value
