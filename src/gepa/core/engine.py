# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import traceback
from dataclasses import dataclass
from typing import Any, Callable, Generic, Literal

from gepa.core.recorder import GEPARecorder
from gepa.core.state import GEPAState, initialize_gepa_state
from gepa.logging.utils import log_detailed_metrics_after_discovering_new_program
from gepa.proposer.merge import MergeProposer
from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer

from .adapter import DataInst, RolloutOutput, Trajectory

# Import tqdm for progress bar functionality
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


@dataclass(frozen=True)
class StepReport:
    iter0: int  # zero-based iteration index
    iter1: int  # human-friendly, iter0 + 1
    action: Literal["reflect", "merge", "skip", "noop", "error"]
    accepted: bool
    new_program_idx: int | None
    reason: str | None
    total_metric_calls: int
    done: bool

    @property
    def i(self) -> int:  # backwards compatibility
        return self.iter1


class GEPAEngine(Generic[DataInst, Trajectory, RolloutOutput]):
    """
    Orchestrates the optimization loop. It uses pluggable ProposeNewCandidate strategies.
    """

    def __init__(
        self,
        run_dir: str | None,
        evaluator: Callable[[list[DataInst], dict[str, str]], tuple[list[RolloutOutput], list[float]]],
        valset: list[DataInst] | None,
        seed_candidate: dict[str, str],
        # Controls
        perfect_score: float,
        seed: int,
        # Strategies and helpers
        reflective_proposer: ReflectiveMutationProposer,
        merge_proposer: MergeProposer | None,
        # Logging
        logger: Any,
        experiment_tracker: Any,
        # Optional parameters
        track_best_outputs: bool = False,
        display_progress_bar: bool = False,
        raise_on_exception: bool = True,
        use_cloudpickle: bool = False,
        # Budget and Stop Condition
        stop_callback: Callable[[Any], bool] | None = None,
    ):
        self.logger = logger
        self.run_dir = run_dir

        # Graceful stopping mechanism
        self._stop_requested = False

        # Set up stopping mechanism
        self.stop_callback = stop_callback
        self.evaluator = evaluator
        self.valset = valset
        self.seed_candidate = seed_candidate

        self.perfect_score = perfect_score
        self.seed = seed
        self.experiment_tracker = experiment_tracker

        self.reflective_proposer = reflective_proposer
        self.merge_proposer = merge_proposer

        self._recorder = GEPARecorder(run_dir, logger=self.logger)
        if self._recorder.enabled:
            self.evaluator = self._recorder.wrap_evaluator(self.evaluator)
        if hasattr(self.reflective_proposer, "attach_recorder"):
            self.reflective_proposer.attach_recorder(self._recorder)
        if self.merge_proposer is not None and hasattr(self.merge_proposer, "attach_recorder"):
            self.merge_proposer.attach_recorder(self._recorder)
        self._dataset_fingerprint: dict[str, str] | None = None

        # Merge scheduling flags (mirroring previous behavior)
        if self.merge_proposer is not None:
            self.merge_proposer.last_iter_found_new_program = False

        self.track_best_outputs = track_best_outputs
        self.display_progress_bar = display_progress_bar
        self.use_cloudpickle = use_cloudpickle

        self.raise_on_exception = raise_on_exception

        # Step API bookkeeping
        self._state: GEPAState | None = None
        self._progress_bar = None
        self._pbar_last_val = 0
        self._started = False
        self._next_stop_decision: bool | None = None
        self._tracker_enabled = True

    def _val_evaluator(self) -> Callable[[dict[str, str]], tuple[list[RolloutOutput], list[float]]]:
        assert self.valset is not None
        return lambda prog: self.evaluator(self.valset, prog)

    def _get_pareto_front_programs(self, state: GEPAState) -> list:
        return state.program_at_pareto_front_valset

    def _run_full_eval_and_add(
        self,
        new_program: dict[str, str],
        state: GEPAState,
        parent_program_idx: list[int],
    ) -> tuple[int, int]:
        num_metric_calls_by_discovery = state.total_num_evals

        if self._recorder.enabled:
            with self._recorder.scope(
                event="EVAL",
                site="full-valset",
                dataset="valset",
                indices=None,
                capture_traces=False,
            ):
                valset_outputs, valset_subscores = self._val_evaluator()(new_program)
        else:
            valset_outputs, valset_subscores = self._val_evaluator()(new_program)
        valset_score = sum(valset_subscores) / len(valset_subscores)

        state.num_full_ds_evals += 1
        state.total_num_evals += len(valset_subscores)

        new_program_idx, linear_pareto_front_program_idx = state.update_state_with_new_program(
            parent_program_idx=parent_program_idx,
            new_program=new_program,
            valset_score=valset_score,
            valset_outputs=valset_outputs,
            valset_subscores=valset_subscores,
            run_dir=self.run_dir,
            num_metric_calls_by_discovery_of_new_program=num_metric_calls_by_discovery,
        )
        state.full_program_trace[-1]["new_program_idx"] = new_program_idx

        if new_program_idx == linear_pareto_front_program_idx:
            self.logger.log(f"Iteration {state.i + 1}: New program is on the linear pareto front")

        log_detailed_metrics_after_discovering_new_program(
            logger=self.logger,
            gepa_state=state,
            valset_score=valset_score,
            new_program_idx=new_program_idx,
            valset_subscores=valset_subscores,
            experiment_tracker=self.experiment_tracker,
            linear_pareto_front_program_idx=linear_pareto_front_program_idx,
        )
        return new_program_idx, linear_pareto_front_program_idx

    def start(self) -> GEPAState:
        """Initialize the optimization run; idempotent."""
        if self._started:
            assert self._state is not None
            return self._state

        self._validate_start_prereqs()
        latest_snapshot = self._configure_recorder_for_start()
        self._setup_progress_bar()
        state = self._initialize_state(latest_snapshot)
        self._state = state
        self._started = True
        return state

    def _validate_start_prereqs(self) -> None:
        if self.valset is None:
            raise ValueError("valset must be provided to GEPAEngine.start()")

    def _configure_recorder_for_start(self) -> dict[str, Any] | None:
        self._next_stop_decision = None
        if not self._recorder.enabled:
            self._tracker_enabled = True
            return None

        latest_snapshot = self._recorder.load_latest_snapshot()
        trainset = getattr(self.reflective_proposer, "trainset", [])
        self._dataset_fingerprint = self._recorder.ensure_dataset_fingerprint(trainset, self.valset)
        next_event_id = latest_snapshot.get("next_event_id", 0) if latest_snapshot else 0
        self._recorder.set_mode("record", next_event_id)
        self._tracker_enabled = not self._recorder.is_replay
        return latest_snapshot

    def _setup_progress_bar(self) -> None:
        if not self.display_progress_bar:
            return

        if tqdm is None:
            self.display_progress_bar = False
            return

        total_calls = None
        if hasattr(self.stop_callback, "max_metric_calls"):
            total_calls = self.stop_callback.max_metric_calls
        elif hasattr(self.stop_callback, "stoppers"):
            for stopper in self.stop_callback.stoppers:
                if hasattr(stopper, "max_metric_calls"):
                    total_calls = stopper.max_metric_calls
                    break

        if total_calls is not None:
            self._progress_bar = tqdm(total=total_calls, desc="GEPA Optimization", unit="rollouts")
        else:
            self._progress_bar = tqdm(desc="GEPA Optimization", unit="rollouts")
        self._progress_bar.update(0)
        self._pbar_last_val = 0

    def _initialize_state(self, latest_snapshot: dict[str, Any] | None) -> GEPAState:
        def _scoped_val_eval(prog: dict[str, str]) -> tuple[list[RolloutOutput], list[float]]:
            if self._recorder.enabled:
                with self._recorder.scope(
                    event="EVAL",
                    site="full-valset",
                    dataset="valset",
                    indices=None,
                    capture_traces=False,
                ):
                    return self._val_evaluator()(prog)
            return self._val_evaluator()(prog)

        state = initialize_gepa_state(
            run_dir=self.run_dir,
            logger=self.logger,
            seed_candidate=self.seed_candidate,
            valset_evaluator=_scoped_val_eval,
            track_best_outputs=self.track_best_outputs,
        )

        assert len(state.pareto_front_valset) == len(self.valset)

        if latest_snapshot is not None:
            self._recorder.restore_from_snapshot(
                latest_snapshot,
                self.reflective_proposer,
                self.merge_proposer,
                self.stop_callback,
            )
            self._next_stop_decision = self._recorder.pop_pending_stop()

        if self._tracker_enabled:
            self.experiment_tracker.log_metrics(
                {
                    "base_program_full_valset_score": state.program_full_scores_val_set[0],
                    "iteration": state.i + 1,
                },
                step=state.i + 1,
            )

        self.logger.log(
            f"Iteration {state.i + 1}: Base program full valset score: {state.program_full_scores_val_set[0]}"
        )

        if self.merge_proposer is not None:
            self.merge_proposer.last_iter_found_new_program = False

        return state

    def step(self) -> StepReport:
        """Execute exactly one iteration body and return a StepReport."""
        state = self.start()
        state = self._state  # mypy hint

        iter0_before = state.i
        iter1_before = iter0_before + 1
        stop_now = self._consume_or_compute_stop(state)
        if stop_now:
            return StepReport(
                iter0=iter0_before,
                iter1=iter1_before,
                action="noop",
                accepted=False,
                new_program_idx=None,
                reason="stop-condition",
                total_metric_calls=state.total_num_evals,
                done=True,
            )

        assert state.is_consistent()
        state.i += 1
        iter0 = state.i
        iter1 = iter0 + 1
        state.full_program_trace.append({"i": iter0})

        def finalize(action: str, accepted: bool, new_idx: int | None, reason: str | None) -> StepReport:
            done_next = self._schedule_next_stop(state)
            self._finalize_iteration(state, iter0, done_next)
            return StepReport(
                iter0=iter0,
                iter1=iter1,
                action=action,
                accepted=accepted,
                new_program_idx=new_idx,
                reason=reason,
                total_metric_calls=state.total_num_evals,
                done=done_next,
            )

        try:
            if self.merge_proposer is not None and self.merge_proposer.use_merge:
                if self.merge_proposer.merges_due > 0 and self.merge_proposer.last_iter_found_new_program:
                    proposal = self.merge_proposer.propose(state)
                    self.merge_proposer.last_iter_found_new_program = False

                    if proposal is not None and proposal.tag == "merge":
                        parent_sums = proposal.subsample_scores_before or [float("-inf"), float("-inf")]
                        new_sum = sum(proposal.subsample_scores_after or [])

                        if new_sum >= max(parent_sums):
                            new_idx, _ = self._run_full_eval_and_add(
                                new_program=proposal.candidate,
                                state=state,
                                parent_program_idx=proposal.parent_program_ids,
                            )
                            self.merge_proposer.merges_due = max(0, self.merge_proposer.merges_due - 1)
                            self.merge_proposer.total_merges_tested += 1
                            return finalize("merge", True, new_idx, "merge-accepted")

                        self.logger.log(
                            f"Iteration {iter1}: New program subsample score {new_sum} is worse than both parents {parent_sums}, skipping merge"
                        )
                        return finalize("merge", False, None, "merge-rejected")

                self.merge_proposer.last_iter_found_new_program = False

            proposal = self.reflective_proposer.propose(state)
            if proposal is None:
                self.logger.log(f"Iteration {iter1}: Reflective mutation did not propose a new candidate")
                return finalize("skip", False, None, "no-proposal")

            old_sum = sum(proposal.subsample_scores_before or [])
            new_sum = sum(proposal.subsample_scores_after or [])
            if new_sum <= old_sum:
                self.logger.log(
                    f"Iteration {iter1}: New subsample score {new_sum} is not better than old score {old_sum}, skipping"
                )
                return finalize("skip", False, None, "not-improved")

            self.logger.log(
                f"Iteration {iter1}: New subsample score {new_sum} is better than old score {old_sum}. Continue to full eval and add to candidate pool."
            )

            new_idx, _ = self._run_full_eval_and_add(
                new_program=proposal.candidate,
                state=state,
                parent_program_idx=proposal.parent_program_ids,
            )

            if self.merge_proposer is not None:
                self.merge_proposer.last_iter_found_new_program = True
                if self.merge_proposer.total_merges_tested < self.merge_proposer.max_merge_invocations:
                    self.merge_proposer.merges_due += 1

            return finalize("reflect", True, new_idx, "reflect-accepted")

        except Exception as exc:  # pylint: disable=broad-except
            self.logger.log(f"Iteration {iter1}: Exception during optimization: {exc}")
            self.logger.log(traceback.format_exc())
            if self.raise_on_exception:
                raise
            return finalize("error", False, None, str(exc))

    def close(self) -> GEPAState:
        """Persist the latest state and close progress resources."""
        if self._state is None:
            raise RuntimeError("GEPAEngine.close() called before start()")

        self._state.save(self.run_dir, use_cloudpickle=self.use_cloudpickle)
        if self.display_progress_bar and self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
        return self._state

    def _finalize_iteration(self, state: GEPAState, iteration: int, pending_stop: bool | None) -> None:
        if self.run_dir is not None:
            state.save(self.run_dir, checkpoint_iter=iteration, use_cloudpickle=self.use_cloudpickle)
        snapshot = self._recorder.capture_snapshot(
            state=state,
            reflective_proposer=self.reflective_proposer,
            merge_proposer=self.merge_proposer,
            stop_callback=self.stop_callback,
            iteration=iteration,
            pending_stop=pending_stop,
        )
        if snapshot:
            self._recorder.persist_snapshot(snapshot, iteration)
        self._update_pbar(state)

    def _update_pbar(self, state: GEPAState) -> None:
        if self.display_progress_bar and self._progress_bar is not None:
            delta = state.total_num_evals - self._pbar_last_val
            if delta > 0:
                self._progress_bar.update(delta)
            self._pbar_last_val = state.total_num_evals

    def seek(self, iteration: int) -> GEPAState:
        """Reload a checkpoint for the given iteration."""
        if self.run_dir is None:
            raise ValueError("seek() requires a run_dir to load checkpoints from")

        snapshot = self._recorder.load_checkpoint_snapshot(iteration) if self._recorder.enabled else None

        state = GEPAState.load_checkpoint(self.run_dir, iteration)
        self._state = state
        self._started = True
        self._stop_requested = False
        self._next_stop_decision = None

        if self._recorder.enabled:
            trainset = getattr(self.reflective_proposer, "trainset", [])
            self._dataset_fingerprint = self._recorder.ensure_dataset_fingerprint(trainset, self.valset)
            pointer = snapshot.get("next_event_id", 0) if snapshot else 0
            mode = "replay" if self._recorder.has_replay_events(pointer) else "record"
            self._recorder.set_mode(mode, pointer)
            if snapshot is not None:
                self._recorder.restore_from_snapshot(
                    snapshot,
                    self.reflective_proposer,
                    self.merge_proposer,
                    self.stop_callback,
                )
                self._next_stop_decision = self._recorder.pop_pending_stop()
            else:
                self._next_stop_decision = None
            if self.merge_proposer is not None:
                self.merge_proposer.last_iter_found_new_program = False
            self._tracker_enabled = mode != "replay"
        else:
            if self.merge_proposer is not None:
                self.merge_proposer.last_iter_found_new_program = False
                if hasattr(self.merge_proposer, "merges_due"):
                    self.merge_proposer.merges_due = 0

        self._update_pbar(state)
        return state

    def run(self) -> GEPAState:
        """Backwards-compatible fire-and-forget loop using the step API."""
        self.start()
        while True:
            report = self.step()
            if report.done:
                break
        return self.close()

    def _consume_or_compute_stop(self, state: GEPAState) -> bool:
        if self._next_stop_decision is not None:
            decision = self._next_stop_decision
            self._next_stop_decision = None
            if self._recorder.enabled:

                def reuse() -> bool:
                    return decision

                recorded = self._recorder.should_stop(reuse, record=True)
                if self._recorder.is_replay and recorded != decision:
                    raise ValueError("STOP decision mismatch during replay")
            return decision
        if self._recorder.enabled:
            return self._recorder.should_stop(lambda: self._compute_stop(state), record=True)
        return self._compute_stop(state)

    def _schedule_next_stop(self, state: GEPAState) -> bool:
        decision = self._compute_stop(state)
        self._next_stop_decision = decision
        return decision

    def _compute_stop(self, state: GEPAState) -> bool:
        if self._stop_requested:
            return True
        if self.stop_callback and self.stop_callback(state):
            return True
        return False

    def request_stop(self):
        """Manually request the optimization to stop gracefully."""
        self.logger.log("Stop requested manually. Initiating graceful shutdown...")
        self._stop_requested = True
