from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from typing import Any

from gepa.core.engine import GEPAEngine
from gepa.core.recorder import GEPARecorder
from gepa.gepa_utils import json_default
from gepa.logging.experiment_tracker import create_experiment_tracker
from gepa.logging.logger import StdOutLogger
from gepa.proposer.base import CandidateProposal
from gepa.utils.stop_condition import MaxTrackedCandidatesStopper


def _evaluator(batch: list[Any], program: dict[str, str]):
    value = int(program["v"])
    outputs = [None for _ in batch]
    scores = [float(value) for _ in batch]
    return outputs, scores


class _TwoStepReflective:
    def propose(self, state) -> CandidateProposal | None:
        iter_idx = state.i
        if iter_idx >= 2:
            return None

        current_parent = iter_idx if iter_idx > 0 else 0
        new_value = iter_idx + 1
        before = [float(new_value - 1), float(new_value - 1)]
        after = [float(new_value), float(new_value)]
        return CandidateProposal(
            candidate={"v": str(new_value)},
            parent_program_ids=[current_parent],
            subsample_indices=[0, 1],
            subsample_scores_before=before,
            subsample_scores_after=after,
            tag="reflective_mutation",
        )


def _make_engine_basic(tmpdir: Path) -> GEPAEngine:
    tracker = create_experiment_tracker(use_wandb=False, use_mlflow=False)
    engine = GEPAEngine(
        run_dir=str(tmpdir),
        evaluator=_evaluator,
        valset=[0, 1, 2, 3],
        seed_candidate={"v": "0"},
        perfect_score=1.0,
        seed=0,
        reflective_proposer=_TwoStepReflective(),
        merge_proposer=None,
        logger=StdOutLogger(),
        experiment_tracker=tracker,
        track_best_outputs=False,
        display_progress_bar=False,
        raise_on_exception=True,
        stop_callback=MaxTrackedCandidatesStopper(3),
    )
    return engine


def _read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _compute_chain(prev_hex: str, payload: dict) -> str:
    dump = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=json_default)
    sha = hashlib.sha256()
    sha.update(prev_hex.encode("utf-8"))
    sha.update(dump.encode("utf-8"))
    return sha.hexdigest()


def test_bit_rot_detected_via_chain(tmp_path):
    run_dir = tmp_path / "bitrot"
    engine = _make_engine_basic(run_dir)
    with engine.experiment_tracker:
        engine.run()

    tape_path = run_dir / "tape" / "events.log"
    events = _read_events(tape_path)
    assert events

    idx = len(events) // 2
    events[idx]["chain_hash"] = "0" * 64  # invalidate the hash
    with tape_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True))
            handle.write("\n")

    engine_replay = _make_engine_basic(run_dir)
    with engine_replay.experiment_tracker:
        with pytest.raises(ValueError, match="chain hash"):
            engine_replay.start()


def test_rewind_and_resume_preserves_chain(tmp_path):
    run_dir = tmp_path / "rewind"
    recorder_initial = GEPARecorder(str(run_dir))
    recorder_initial._append_event({"type": "STOP", "decision": False})
    recorder_initial._append_event({"type": "STOP", "decision": True})

    recorder_truncated = GEPARecorder(str(run_dir))
    recorder_truncated.set_mode("record", 1)
    recorder_truncated._append_event({"type": "STOP", "decision": False})

    tape_path = run_dir / "tape" / "events.log"
    events = _read_events(tape_path)
    assert len(events) == 2

    prev = "0" * 64
    for event in events:
        assert "chain_prev" in event and "chain_hash" in event and "event_id" in event
        assert event["chain_prev"] == prev
        payload = {k: v for k, v in event.items() if k != "chain_hash"}
        assert event["chain_hash"] == _compute_chain(prev, payload)
        prev = event["chain_hash"]


def test_dataset_fingerprint_strict_guard(tmp_path, monkeypatch):
    monkeypatch.setenv("GEPA_RECORDER_STRICT", "1")
    run_dir = tmp_path / "strict"

    recorder_first = GEPARecorder(str(run_dir))
    recorder_first.ensure_dataset_fingerprint([0, 1, 2], [10, 11])

    recorder_second = GEPARecorder(str(run_dir))
    with pytest.raises(ValueError, match="Dataset fingerprint mismatch"):
        recorder_second.ensure_dataset_fingerprint([2, 1, 0], [10, 11])
