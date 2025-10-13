from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gepa.core.engine import GEPAEngine
from gepa.core.result import GEPAResult
from gepa.core.state import GEPAState
from gepa.logging.experiment_tracker import create_experiment_tracker
from gepa.logging.logger import StdOutLogger
from gepa.proposer.base import CandidateProposal
from gepa.utils.stop_condition import MaxTrackedCandidatesStopper


def _evaluator(batch: list[Any], program: dict[str, str]):
    """Deterministic evaluator: score equals the integer held in candidate['v']."""
    value = int(program["v"])
    outputs = [None for _ in batch]
    scores = [float(value) for _ in batch]
    return outputs, scores


class TwoStepReflective:
    """Proposer that emits two strictly improving candidates, then stops."""

    def propose(self, state: GEPAState) -> CandidateProposal | None:
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


def _make_engine(tmpdir: Path) -> GEPAEngine:
    tracker = create_experiment_tracker(use_wandb=False, use_mlflow=False)
    engine = GEPAEngine(
        run_dir=str(tmpdir),
        evaluator=_evaluator,
        valset=[0, 1, 2, 3],
        seed_candidate={"v": "0"},
        perfect_score=1.0,
        seed=0,
        reflective_proposer=TwoStepReflective(),
        merge_proposer=None,
        logger=StdOutLogger(),
        experiment_tracker=tracker,
        track_best_outputs=False,
        display_progress_bar=False,
        raise_on_exception=True,
        stop_callback=MaxTrackedCandidatesStopper(3),
    )
    return engine


def test_step_api_matches_run_and_seek(tmp_path):
    run_full_dir = tmp_path / "run_full"
    engine_full = _make_engine(run_full_dir)
    with engine_full.experiment_tracker:
        state_full = engine_full.run()
    result_full = GEPAResult.from_state(state_full)
    assert result_full.num_candidates == 3
    assert result_full.val_aggregate_scores[-1] > result_full.val_aggregate_scores[0]

    run_step_dir = tmp_path / "run_step"
    engine_step = _make_engine(run_step_dir)
    reports = []
    with engine_step.experiment_tracker:
        engine_step.start()
        while True:
            report = engine_step.step()
            reports.append(report)
            if report.done:
                break
        state_step = engine_step.close()
    result_step = GEPAResult.from_state(state_step)
    assert result_step.num_candidates == 3
    assert [round(s, 6) for s in result_full.val_aggregate_scores] == [
        round(s, 6) for s in result_step.val_aggregate_scores
    ]
    assert any(r.action == "reflect" and r.accepted for r in reports)

    latest_snapshot_path = run_step_dir / "control_state_latest.json"
    assert latest_snapshot_path.exists()
    latest_snapshot = json.loads(latest_snapshot_path.read_text())
    for required_key in ["iter", "next_event_id", "rng", "merge", "stoppers"]:
        assert required_key in latest_snapshot

    tape_path = run_step_dir / "tape" / "events.log"
    assert tape_path.exists()
    events = [json.loads(line) for line in tape_path.read_text().splitlines() if line.strip()]
    assert events
    type_counts: dict[str, int] = {"EVAL": 0, "PROPOSE": 0, "STOP": 0}
    for event in events:
        etype = event["type"]
        if etype in type_counts:
            type_counts[etype] += 1
    assert type_counts["EVAL"] > 0
    assert type_counts["STOP"] > 0
    next_event_id = latest_snapshot["next_event_id"]
    assert events[-1]["event_id"] == next_event_id - 1

    checkpoint_dir = run_step_dir / "checkpoints"
    assert checkpoint_dir.exists()
    checkpoint_files = sorted(checkpoint_dir.glob("iter_*.json"))
    assert checkpoint_files, "expected checkpoint control metadata files"
    checkpoint_snapshot = json.loads(checkpoint_files[-1].read_text())
    assert checkpoint_snapshot["next_event_id"] <= next_event_id

    run_seek_dir = tmp_path / "run_seek"
    engine_seek = _make_engine(run_seek_dir)
    with engine_seek.experiment_tracker:
        engine_seek.start()
        first_report = engine_seek.step()
        assert first_report.i == 1
        engine_seek.seek(0)
        finished = False
        while True:
            rep = engine_seek.step()
            if rep.done:
                finished = True
                break
        assert finished
        state_seek = engine_seek.close()
    result_seek = GEPAResult.from_state(state_seek)
    assert result_seek.num_candidates == 3
    assert [round(s, 6) for s in result_full.val_aggregate_scores] == [
        round(s, 6) for s in result_seek.val_aggregate_scores
    ]
