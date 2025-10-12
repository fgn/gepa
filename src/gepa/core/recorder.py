from __future__ import annotations

import base64
import hashlib
import json
import pickle
import random
import warnings
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from gepa.gepa_utils import json_default
from gepa.utils.stop_condition import CompositeStopper, NoImprovementStopper, StopperProtocol

EventType = Literal["EVAL", "PROPOSE", "STOP"]


def _encode_random_state(rng: random.Random) -> str:
    return base64.b64encode(pickle.dumps(rng.getstate())).decode("utf-8")


def _decode_random_state(rng: random.Random, payload: str) -> None:
    rng.setstate(pickle.loads(base64.b64decode(payload.encode("utf-8"))))


def _serialize(obj: Any) -> str:
    return base64.b64encode(pickle.dumps(obj)).decode("utf-8")


def _deserialize(payload: str) -> Any:
    return pickle.loads(base64.b64decode(payload.encode("utf-8")))


def _candidate_hash(candidate: Mapping[str, Any]) -> str:
    canon = json.dumps(candidate, sort_keys=True, separators=(",", ":"), default=json_default)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _dataset_hash(dataset: Mapping[str, Any]) -> str:
    canon = json.dumps(dataset, sort_keys=True, default=json_default)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _fingerprint_sequence(seq: Sequence[Any], label: str) -> str:
    sha = hashlib.sha256()
    sha.update(label.encode("utf-8"))
    size = len(seq)
    sha.update(str(size).encode("utf-8"))
    sample = list(seq[: min(size, 5)])
    tail = list(seq[-5:]) if size > 5 else []

    def _update(prefix: str, item: Any) -> None:
        try:
            payload = json.dumps(item, sort_keys=True, default=json_default)
        except TypeError:
            payload = repr(item)
        sha.update(prefix.encode("utf-8"))
        sha.update(payload.encode("utf-8"))

    for item in sample:
        _update("head", item)
    for item in tail:
        _update("tail", item)
    return f"sha256:{sha.hexdigest()}"


def compute_dataset_fingerprint(trainset: Sequence[Any], valset: Sequence[Any] | None) -> dict[str, str]:
    fingerprint = {"trainset": _fingerprint_sequence(trainset, "trainset")}
    if valset is not None:
        fingerprint["valset"] = _fingerprint_sequence(valset, "valset")
    return fingerprint


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


@dataclass(frozen=True)
class _Scope:
    event: EventType
    site: str | None
    dataset: str | None
    indices: list[int] | None
    capture_traces: bool
    metadata: dict[str, Any]


class GEPARecorder:
    """Recorder/tape orchestrator providing deterministic replay."""

    def __init__(
        self,
        run_dir: str | None,
        *,
        strict_dataset_check: bool = False,
        logger: Any | None = None,
    ):
        self.enabled = run_dir is not None
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.strict_dataset_check = strict_dataset_check
        self.logger = logger

        self.mode: Literal["record", "replay"] = "record"
        self.next_event_id: int = 0
        self._replay_events: list[dict[str, Any]] = []
        self._replay_index: int = 0
        self._current_scope: _Scope | None = None
        self.dataset_fingerprint: dict[str, str] | None = None
        self._pending_stop_from_snapshot: bool | None = None

        if self.enabled:
            _ensure_dir(self.run_dir / "tape")
            self._events_path = self.run_dir / "tape" / "events.log"
            self._dataset_fingerprint_path = self.run_dir / "dataset_fingerprint.json"
        else:
            self._events_path = None
            self._dataset_fingerprint_path = None

    # --------------------------------------------------------------------- public API
    @property
    def is_replay(self) -> bool:
        return self.enabled and self.mode == "replay"

    @contextmanager
    def scope(
        self,
        *,
        event: EventType,
        site: str | None,
        dataset: str | None,
        indices: Sequence[int] | None,
        capture_traces: bool,
        metadata: Mapping[str, Any] | None = None,
    ):
        """Annotate the next adapter/evaluator/LM calls within this block."""
        normalized_indices = list(indices) if indices is not None else None
        scope = _Scope(
            event=event,
            site=site,
            dataset=dataset,
            indices=normalized_indices,
            capture_traces=capture_traces,
            metadata=dict(metadata or {}),
        )
        previous = self._current_scope
        self._current_scope = scope
        try:
            yield
        finally:
            self._current_scope = previous

    def pop_pending_stop(self) -> bool | None:
        value = self._pending_stop_from_snapshot
        self._pending_stop_from_snapshot = None
        return value

    def wrap_adapter(self, adapter: GEPAAdapter) -> GEPAAdapter:
        if not self.enabled or getattr(adapter, "_gepa_recorder_wrapped", False):
            return adapter

        recorder = self

        class RecordingAdapter:
            _gepa_recorder_wrapped = True

            def __init__(self, inner: GEPAAdapter):
                self._inner = inner

            def evaluate(
                self,
                batch: list[Any],
                candidate: dict[str, str],
                capture_traces: bool = False,
            ) -> EvaluationBatch:
                return recorder._handle_evaluate(
                    call=lambda: self._inner.evaluate(batch, candidate, capture_traces=capture_traces),
                    candidate=candidate,
                    expect_batch=True,
                    capture_traces=capture_traces,
                )

            def __getattr__(self, item):
                return getattr(self._inner, item)

        return RecordingAdapter(adapter)

    def wrap_evaluator(
        self,
        evaluator: Callable[[list[Any], dict[str, str]], tuple[list[Any], list[float]]],
    ) -> Callable[[list[Any], dict[str, str]], tuple[list[Any], list[float]]]:
        if not self.enabled or getattr(evaluator, "_gepa_recorder_wrapped", False):
            return evaluator

        recorder = self

        def wrapped(batch: list[Any], candidate: dict[str, str]) -> tuple[list[Any], list[float]]:
            return recorder._handle_evaluate(
                call=lambda: evaluator(batch, candidate),
                candidate=candidate,
                expect_batch=False,
                capture_traces=False,
            )

        wrapped._gepa_recorder_wrapped = True
        return wrapped

    def wrap_lm(self, lm: Callable[..., Any] | None) -> Callable[..., Any] | None:
        if not (self.enabled and lm is not None):
            return lm
        if getattr(lm, "_gepa_recorder_wrapped", False):
            return lm

        recorder = self

        def wrapped(*args, **kwargs):
            scope = recorder._require_scope("PROPOSE", allow_none=True)
            if scope is None or recorder.mode == "record":
                return lm(*args, **kwargs)
            event = recorder._consume_event("PROPOSE")
            return _deserialize(event["new_texts"])

        wrapped._gepa_recorder_wrapped = True
        return wrapped

    def record_proposal(
        self,
        *,
        iteration: int,
        parent_prog_id: int,
        components: list[str],
        reflective_dataset: Mapping[str, Any],
        new_texts: Mapping[str, str],
    ) -> None:
        if not (self.enabled and self.mode == "record"):
            return
        event = {
            "type": "PROPOSE",
            "iter": iteration,
            "parent_prog_id": parent_prog_id,
            "components": components,
            "reflective_dataset_hash": _dataset_hash(reflective_dataset),
            "new_texts": _serialize(dict(new_texts)),
        }
        self._append_event(event)

    def replay_proposal(
        self,
        *,
        parent_prog_id: int,
        components: list[str],
        reflective_dataset: Mapping[str, Any],
    ) -> dict[str, str]:
        if not self.enabled:
            raise RuntimeError("Replay requested without recorder enabled")
        event = self._consume_event("PROPOSE")
        if event["parent_prog_id"] != parent_prog_id:
            raise ValueError(
                f"Recorded parent program {event['parent_prog_id']} does not match requested {parent_prog_id}"
            )
        if event["components"] != components:
            raise ValueError("Recorded proposal components do not match requested components")
        expected_hash = event["reflective_dataset_hash"]
        actual_hash = _dataset_hash(reflective_dataset)
        if expected_hash != actual_hash:
            raise ValueError("Reflective dataset hash mismatch during replay")
        return _deserialize(event["new_texts"])

    def should_stop(self, compute_fn: Callable[[], bool], *, record: bool = True) -> bool:
        if not self.enabled:
            return compute_fn()
        if self.mode == "record":
            decision = compute_fn()
            if record:
                self._append_event({"type": "STOP", "decision": decision})
            return decision
        if record:
            event = self._consume_event("STOP")
            return bool(event["decision"])
        if not self._replay_events:
            self._load_replay_events()
        if self._replay_index >= len(self._replay_events):
            raise IndexError("Replay exhausted while peeking STOP")
        event = self._replay_events[self._replay_index]
        if event.get("type") != "STOP":
            raise ValueError("Expected STOP event while peeking")
        return bool(event["decision"])

    # ------------------------------------------------------------------ snapshot API
    def capture_snapshot(
        self,
        *,
        state: Any,
        reflective_proposer: Any,
        merge_proposer: Any | None,
        stop_callback: StopperProtocol | None,
        iteration: int,
        pending_stop: bool | None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}

        rng_state: dict[str, Any] = {}
        selector = getattr(reflective_proposer, "candidate_selector", None)
        if selector is not None and hasattr(selector, "rng") and isinstance(selector.rng, random.Random):
            rng_state["candidate_selector"] = _encode_random_state(selector.rng)

        sampler = getattr(reflective_proposer, "batch_sampler", None)
        if sampler is not None and hasattr(sampler, "rng") and isinstance(sampler.rng, random.Random):
            rng_state["batch_sampler"] = {
                "random": _encode_random_state(sampler.rng),
                "epoch": getattr(sampler, "epoch", -1),
                "shuffled_ids": list(getattr(sampler, "shuffled_ids", [])),
                "id_freqs": [(int(k), int(v)) for k, v in getattr(sampler, "id_freqs", Counter()).items()],
            }

        adapter = getattr(reflective_proposer, "adapter", None)
        if adapter is not None and hasattr(adapter, "rng") and isinstance(adapter.rng, random.Random):
            rng_state.setdefault("adapter", {})["dspy"] = _encode_random_state(adapter.rng)

        merge_state: dict[str, Any] = {}
        if merge_proposer is not None:
            if hasattr(merge_proposer, "rng") and isinstance(merge_proposer.rng, random.Random):
                rng_state["merge_proposer"] = _encode_random_state(merge_proposer.rng)
            merge_state = {
                "merges_due": getattr(merge_proposer, "merges_due", 0),
                "total_merges_tested": getattr(merge_proposer, "total_merges_tested", 0),
                "last_iter_found_new_program": getattr(merge_proposer, "last_iter_found_new_program", False),
                "merges_performed": {
                    "triplets": [list(t) for t in getattr(merge_proposer, "merges_performed", ([], []))[0]],
                    "desc_triplets": [
                        [t[0], t[1], list(t[2])] for t in getattr(merge_proposer, "merges_performed", ([], []))[1]
                    ],
                },
            }

        stopper_state: dict[str, Any] = {}
        if stop_callback is not None:
            self._collect_stop_state(stop_callback, stopper_state)

        return {
            "iter": iteration,
            "next_event_id": self.next_event_id,
            "rng": rng_state,
            "merge": merge_state,
            "stoppers": stopper_state,
            "pending_stop": pending_stop,
        }

    def persist_snapshot(self, snapshot: Mapping[str, Any], iteration: int) -> None:
        if not self.enabled:
            return
        latest_path = self.run_dir / "control_state_latest.json"
        _write_json_atomic(latest_path, snapshot)
        ckpt_dir = self.run_dir / "checkpoints"
        _ensure_dir(ckpt_dir)
        ckpt_path = ckpt_dir / f"iter_{iteration}.json"
        _write_json_atomic(ckpt_path, snapshot)

    def load_latest_snapshot(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self.run_dir / "control_state_latest.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_checkpoint_snapshot(self, iteration: int) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("Recorder not enabled; cannot load snapshot")
        ckpt_path = self.run_dir / "checkpoints" / f"iter_{iteration}.json"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint snapshot not found for iteration {iteration} at {ckpt_path}")
        with ckpt_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def restore_from_snapshot(
        self,
        snapshot: Mapping[str, Any],
        reflective_proposer: Any,
        merge_proposer: Any | None,
        stop_callback: StopperProtocol | None,
    ) -> None:
        if not self.enabled:
            return
        self.next_event_id = int(snapshot.get("next_event_id", 0))
        self._restore_rng_state(snapshot.get("rng", {}), reflective_proposer, merge_proposer)
        if stop_callback is not None:
            self._restore_stop_state(stop_callback, snapshot.get("stoppers", {}))
        self._pending_stop_from_snapshot = snapshot.get("pending_stop")
        if merge_proposer is not None:
            merge_state = snapshot.get("merge", {})
            merge_proposer.merges_due = int(merge_state.get("merges_due", 0))
            merge_proposer.total_merges_tested = int(merge_state.get("total_merges_tested", 0))
            merge_proposer.last_iter_found_new_program = bool(merge_state.get("last_iter_found_new_program", False))
            recorded = merge_state.get("merges_performed", {"triplets": [], "desc_triplets": []})
            triplets = [tuple(t) for t in recorded.get("triplets", [])]
            desc = [(t[0], t[1], tuple(t[2])) for t in recorded.get("desc_triplets", [])]
            merge_proposer.merges_performed = (triplets, desc)

    # ---------------------------------------------------------------- dataset guard
    def ensure_dataset_fingerprint(self, trainset: Sequence[Any], valset: Sequence[Any] | None) -> dict[str, str]:
        fingerprint = compute_dataset_fingerprint(trainset, valset)
        if not self.enabled:
            self.dataset_fingerprint = fingerprint
            return fingerprint

        if self.dataset_fingerprint is None:
            if self._dataset_fingerprint_path.exists():
                with self._dataset_fingerprint_path.open("r", encoding="utf-8") as handle:
                    recorded = json.load(handle)
                self.dataset_fingerprint = recorded
                if fingerprint != recorded:
                    self._handle_dataset_mismatch(fingerprint, recorded)
            else:
                self.dataset_fingerprint = fingerprint
                _write_json_atomic(self._dataset_fingerprint_path, fingerprint)
        else:
            if fingerprint != self.dataset_fingerprint:
                self._handle_dataset_mismatch(fingerprint, self.dataset_fingerprint)
        return self.dataset_fingerprint

    def has_replay_events(self, pointer: int) -> bool:
        if not self.enabled or not self._events_path.exists():
            return False
        with self._events_path.open("r", encoding="utf-8") as handle:
            total_events = sum(1 for _ in handle)
        return total_events > pointer

    def set_mode(self, mode: Literal["record", "replay"], pointer: int | None = None) -> None:
        if not self.enabled:
            return
        pointer = 0 if pointer is None else int(pointer)
        self.mode = mode
        if mode == "record":
            self._prepare_record_mode(pointer)
        else:
            self._prepare_replay_mode(pointer)

    # ------------------------------------------------------------------ internal helpers
    def _handle_evaluate(
        self,
        *,
        call: Callable[[], Any],
        candidate: Mapping[str, Any],
        expect_batch: bool,
        capture_traces: bool,
    ) -> Any:
        scope = self._require_scope("EVAL", allow_none=not self.enabled)
        if scope is None or not self.enabled:
            return call()

        if self.mode == "record":
            result = call()
            if expect_batch and not isinstance(result, EvaluationBatch):
                raise TypeError("Adapter evaluate must return EvaluationBatch when recorder is active")
            outputs: list[Any]
            scores: list[float]
            trajectories: list[Any] | None
            if expect_batch:
                outputs = list(result.outputs)
                scores = list(result.scores)
                trajectories = list(result.trajectories) if result.trajectories is not None else None
            else:
                outputs, scores = result
                outputs = list(outputs)
                scores = list(scores)
                trajectories = None
            event = {
                "type": "EVAL",
                "event_metadata": {
                    "site": scope.site,
                    "dataset": scope.dataset,
                    "indices": scope.indices,
                    "capture_traces": capture_traces,
                },
                "candidate_hash": _candidate_hash(candidate),
                "result_kind": "batch" if expect_batch else "tuple",
                "outputs": _serialize(outputs),
                "scores": _serialize(scores),
                "trajectories": _serialize(trajectories) if trajectories is not None else None,
            }
            self._append_event(event)
            return result

        event = self._consume_event("EVAL")
        meta = event["event_metadata"]
        self._validate_scope(scope, meta)
        if event["candidate_hash"] != _candidate_hash(candidate):
            raise ValueError("Candidate hash mismatch during replay")
        outputs = _deserialize(event["outputs"])
        scores = _deserialize(event["scores"])
        trajectories = _deserialize(event["trajectories"]) if event["trajectories"] is not None else None
        if expect_batch:
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)
        return (outputs, scores)

    def _require_scope(self, expected_event: EventType, *, allow_none: bool) -> _Scope | None:
        scope = self._current_scope
        if scope is None:
            if allow_none:
                return None
            raise RuntimeError(f"Recorder scope is required for {expected_event} events")
        if scope.event != expected_event:
            raise RuntimeError(f"Recorder scope event mismatch: expected {expected_event}, found {scope.event}")
        return scope

    def _validate_scope(self, scope: _Scope, meta: Mapping[str, Any]) -> None:
        if scope.site != meta.get("site"):
            raise ValueError(f"Site mismatch during replay: expected {scope.site}, got {meta.get('site')}")
        if scope.dataset != meta.get("dataset"):
            raise ValueError(f"Dataset mismatch during replay: expected {scope.dataset}, got {meta.get('dataset')}")
        if scope.indices != meta.get("indices"):
            raise ValueError("Indices mismatch during replay")
        if scope.capture_traces != meta.get("capture_traces"):
            raise ValueError("capture_traces flag mismatch during replay")

    def _append_event(self, event: Mapping[str, Any]) -> None:
        event_payload = dict(event)
        event_payload["event_id"] = self.next_event_id
        with self._events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event_payload, sort_keys=True))
            handle.write("\n")
        self.next_event_id += 1

    def _consume_event(self, expected_type: EventType) -> dict[str, Any]:
        if not self._replay_events:
            self._load_replay_events()
        if self._replay_index >= len(self._replay_events):
            raise IndexError(f"Replay exhausted. Expected event type {expected_type} at id {self.next_event_id}.")
        event = self._replay_events[self._replay_index]
        if event.get("type") != expected_type:
            raise ValueError(
                f"Replay event mismatch: expected {expected_type}, found {event.get('type')} at id {event.get('event_id')}"
            )
        if event.get("event_id") != self.next_event_id:
            raise ValueError(f"Replay event id mismatch: expected {self.next_event_id}, found {event.get('event_id')}")
        self._replay_index += 1
        self.next_event_id += 1
        return event

    def _load_replay_events(self) -> None:
        if not self._events_path.exists():
            raise FileNotFoundError("Replay requested but no events.log found")
        with self._events_path.open("r", encoding="utf-8") as handle:
            events = [json.loads(line) for line in handle if line.strip()]
        self._replay_events = events
        self._replay_index = 0

    def _prepare_record_mode(self, pointer: int) -> None:
        self._pending_stop_from_snapshot = None
        if not self._events_path.exists():
            with self._events_path.open("w", encoding="utf-8"):
                pass
            self.next_event_id = 0
            return
        with self._events_path.open("r", encoding="utf-8") as handle:
            existing = [json.loads(line) for line in handle if line.strip()]
        if pointer > len(existing):
            raise ValueError("Pointer exceeds number of recorded events")
        if pointer < len(existing):
            # truncate to pointer
            remaining = existing[:pointer]
            with self._events_path.open("w", encoding="utf-8") as handle:
                for event in remaining:
                    handle.write(json.dumps(event, sort_keys=True))
                    handle.write("\n")
        self.next_event_id = pointer

    def _prepare_replay_mode(self, pointer: int) -> None:
        self._pending_stop_from_snapshot = None
        self._load_replay_events()
        if pointer > len(self._replay_events):
            raise ValueError("Pointer exceeds number of recorded events")
        self._replay_index = pointer
        self.next_event_id = pointer

    def _restore_rng_state(
        self,
        rng_payload: Mapping[str, Any],
        reflective_proposer: Any,
        merge_proposer: Any | None,
    ) -> None:
        selector_state = rng_payload.get("candidate_selector")
        if selector_state and hasattr(reflective_proposer.candidate_selector, "rng"):
            _decode_random_state(reflective_proposer.candidate_selector.rng, selector_state)

        batch_payload = rng_payload.get("batch_sampler", {})
        if batch_payload and hasattr(reflective_proposer.batch_sampler, "rng"):
            sampler = reflective_proposer.batch_sampler
            random_state = batch_payload.get("random")
            if random_state:
                _decode_random_state(sampler.rng, random_state)
            sampler.epoch = int(batch_payload.get("epoch", -1))
            sampler.shuffled_ids = list(batch_payload.get("shuffled_ids", []))
            sampler.id_freqs = Counter({int(k): int(v) for k, v in batch_payload.get("id_freqs", [])})

        adapter_state = rng_payload.get("adapter", {})
        if adapter_state and hasattr(reflective_proposer.adapter, "rng"):
            dspy_state = adapter_state.get("dspy")
            if dspy_state:
                _decode_random_state(reflective_proposer.adapter.rng, dspy_state)

        if merge_proposer is not None:
            merge_state = rng_payload.get("merge_proposer")
            if merge_state and hasattr(merge_proposer, "rng"):
                _decode_random_state(merge_proposer.rng, merge_state)

    def _collect_stop_state(self, stopper: StopperProtocol, store: dict[str, Any]) -> None:
        if isinstance(stopper, NoImprovementStopper):
            store["NoImprovementStopper"] = {
                "iterations_without_improvement": stopper.iterations_without_improvement,
                "best_score": stopper.best_score,
            }
        elif isinstance(stopper, CompositeStopper):
            for child in stopper.stoppers:
                self._collect_stop_state(child, store)

    def _restore_stop_state(self, stopper: StopperProtocol, payload: Mapping[str, Any]) -> None:
        if isinstance(stopper, NoImprovementStopper):
            state = payload.get("NoImprovementStopper")
            if state:
                stopper.iterations_without_improvement = int(state["iterations_without_improvement"])
                stopper.best_score = float(state["best_score"])
        elif isinstance(stopper, CompositeStopper):
            for child in stopper.stoppers:
                self._restore_stop_state(child, payload)

    def _handle_dataset_mismatch(self, current: dict[str, str], recorded: dict[str, str]) -> None:
        message = (
            "Dataset fingerprint mismatch detected. "
            "The trainset/valset ordering or contents differ from the recorded run."
        )
        if self.strict_dataset_check:
            raise ValueError(message)
        if self.logger is not None and hasattr(self.logger, "log"):
            self.logger.log(f"WARNING: {message}")
        else:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
