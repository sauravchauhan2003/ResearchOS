from ui.action_log import ActionLog


def test_action_log_updates_steps_and_renders_action_snapshots() -> None:
    rendered = []
    actions = ActionLog(rendered.append)
    actions.emit("Searching <web>", "running")
    actions.emit("Searching <web>", "complete", "Found 2 sources")

    assert len(actions.snapshot()) == 1
    assert actions.snapshot()[0]["status"] == "complete"
    assert actions.snapshot()[0]["detail"] == "Found 2 sources"
    assert rendered[-1][0]["action"] == "Searching <web>"
    assert rendered[-1][0]["detail"] == "Found 2 sources"
