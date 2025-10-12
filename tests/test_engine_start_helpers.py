from pathlib import Path

from gepa.core.adapter import EvaluationBatch
from gepa.core.engine import GEPAEngine
from gepa.core.state import GEPAState
from gepa.logging.experiment_tracker import create_experiment_tracker
from gepa.logging.logger import StdOutLogger
from gepa.proposer.base import CandidateProposal
from gepa.proposer.reflective_mutation.base import BatchSampler, CandidateSelector, ReflectionComponentSelector
from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer


def _stub_evaluator(batch, program):
    outputs = [None for _ in batch]
    scores = [float(int(program["v"])) for _ in batch]
    return outputs, scores


class _StubSelector(CandidateSelector):
    def select_candidate_idx(self, state: GEPAState) -> int:
        return len(state.program_candidates) - 1


class _StubBatchSampler(BatchSampler):
    def next_minibatch_indices(self, trainset_size: int, iteration: int) -> list[int]:
        return list(range(min(trainset_size, 2)))


class _StubComponentSelector(ReflectionComponentSelector):
    def __call__(self, *_args, **_kwargs):
        return ["v"]


class _StubAdapter:
    def evaluate(self, batch, candidate, capture_traces=False):
        scores = [float(int(candidate["v"])) for _ in batch]
        trajectories = [[] for _ in batch] if capture_traces else None
        return EvaluationBatch(outputs=[None for _ in batch], scores=scores, trajectories=trajectories)

    def make_reflective_dataset(self, *_args, **_kwargs):
        return {"v": [{"delta": 1}]}


class _StubReflective(ReflectiveMutationProposer):
    def __init__(self):
        super().__init__(
            logger=StdOutLogger(),
            trainset=[0, 1, 2],
            adapter=_StubAdapter(),
            candidate_selector=_StubSelector(),
            module_selector=_StubComponentSelector(),
            batch_sampler=_StubBatchSampler(),
            perfect_score=10.0,
            skip_perfect_score=False,
            experiment_tracker=create_experiment_tracker(use_wandb=False, use_mlflow=False),
        )

    def propose(self, state: GEPAState) -> CandidateProposal | None:
        value = state.i + 1
        if value >= 2:
            return None
        return CandidateProposal(
            candidate={"v": str(value + 1)},
            parent_program_ids=[0],
            subsample_indices=[0, 1],
            subsample_scores_before=[float(value), float(value)],
            subsample_scores_after=[float(value + 1), float(value + 1)],
            tag="reflective_mutation",
        )


def _make_engine(run_dir: Path | None) -> GEPAEngine:
    tracker = create_experiment_tracker(use_wandb=False, use_mlflow=False)
    return GEPAEngine(
        run_dir=str(run_dir) if run_dir is not None else None,
        evaluator=_stub_evaluator,
        valset=[0, 1, 2, 3],
        seed_candidate={"v": "0"},
        perfect_score=1.0,
        seed=0,
        reflective_proposer=_StubReflective(),
        merge_proposer=None,
        logger=StdOutLogger(),
        experiment_tracker=tracker,
        track_best_outputs=False,
        display_progress_bar=False,
        raise_on_exception=True,
        stop_callback=None,
    )


def test_configure_recorder_no_run_dir():
    engine = _make_engine(None)
    snapshot = engine._configure_recorder_for_start()
    assert snapshot is None
    assert engine._tracker_enabled is True


def test_configure_recorder_with_run_dir(tmp_path):
    engine = _make_engine(tmp_path)
    snapshot = engine._configure_recorder_for_start()
    assert snapshot is None
    assert engine._dataset_fingerprint is not None
    assert engine._recorder.mode == "record"
    fingerprint_file = tmp_path / "dataset_fingerprint.json"
    assert fingerprint_file.exists()


def test_initialize_state_returns_state(tmp_path):
    engine = _make_engine(tmp_path)
    engine._configure_recorder_for_start()
    state = engine._initialize_state(None)
    assert isinstance(state, GEPAState)
    assert state.i == -1
