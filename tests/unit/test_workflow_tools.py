from __future__ import annotations

import json

from scenesmith.agent_utils.workflow_tools import WorkflowTools


def test_checkpoint_reset_invalidates_pending_todos_without_reusing_coordinates() -> None:
    workflow = WorkflowTools()
    workflow._designer_todo_manager_impl("add", "Move dining table to x=0, y=1.2")
    workflow._designer_todo_manager_impl("add", "Verify chair clearance")
    workflow._designer_todo_manager_impl("complete")

    invalidated = workflow.invalidate_pending_todos("scene restored; re-read poses")

    assert invalidated == 1
    payload = json.loads(
        workflow._designer_todo_manager_impl("list_all")
    )
    by_task = {todo["task"]: todo for todo in payload["todos"]}
    assert by_task["Move dining table to x=0, y=1.2"]["status"] == "invalidated"
    assert (
        by_task["Move dining table to x=0, y=1.2"]["invalidation_reason"]
        == "scene restored; re-read poses"
    )
    assert by_task["Verify chair clearance"]["status"] == "completed"
