from test_step_api import _make_engine

from gepa.core.result import GEPAResult


def test_recorder_replay_from_checkpoint(tmp_path):
    run_dir = tmp_path / "run_record"
    engine = _make_engine(run_dir)
    with engine.experiment_tracker:
        engine.start()
        while True:
            report = engine.step()
            if report.done:
                break
        state_final = engine.close()

    original = GEPAResult.from_state(state_final)

    # Replay from checkpoint after first iteration
    engine_replay = _make_engine(run_dir)
    with engine_replay.experiment_tracker:
        engine_replay.start()
        engine_replay.seek(0)
        while True:
            report = engine_replay.step()
            if report.done:
                break
        replay_state = engine_replay.close()

    replay = GEPAResult.from_state(replay_state)

    assert replay.num_candidates == original.num_candidates
    assert [round(s, 6) for s in replay.val_aggregate_scores] == [round(s, 6) for s in original.val_aggregate_scores]
