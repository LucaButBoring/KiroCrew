"""Workflow nesting caps on tasks that ``task_run`` starts from inside a run.

``task_run`` posts to ``/api/taskrunner``, and the task runner registers the
task's shared run itself. ``api_taskrunner_start`` checks the caller against the
caps before the task starts, so a chain of task runs cannot restart the count.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from handlers_taskrunner_coverage_helpers import (  # noqa: F401 -- the autouse fixture registers by import
    _body,
    _request,
    _runner,
    _stub_sel,
)

from kiro_crew.dashboard.handlers.taskrunner import api_taskrunner_start
from kiro_crew.workflows.nesting import ERROR_CODE_DEPTH, ERROR_CODE_FANOUT
from kiro_crew.workflows.service import WorkflowService

QUICK = (
    'META = {"name": "quick", "description": "d"}\n'
    "async def workflow(ctx):\n"
    "    return {'ok': True}\n"
)

# Turns a lost run into a failed test well under pytest-timeout; never a barrier.
_HANG_GUARD_SECS = 30.0


class _Sessions:
    async def get_or_create(self, key, **kw):
        return SimpleNamespace(), True, False

    def release(self, key, *, cleanup=False):
        pass

    async def destroy(self, key):
        pass


def _wired(tmp_path, **caps):
    """A task runner mock whose start registers a shared run, as TaskRunner does."""
    svc = WorkflowService(sessions=_Sessions(), persist=False, pool_agents=False, **caps)
    runner = _runner(tmp_path)

    async def start_background(spec_path, *, session_key="", **_kw):
        task_id = f"t{len(runner._runs) + 1}"
        run_id = await svc.begin_host_run(
            name=task_id, source_format="task-plan", driver="taskrunner", session_key=session_key
        )
        runner._runs[task_id] = SimpleNamespace(workflow_run_id=run_id, spec_path=str(spec_path))
        return task_id

    runner.start_background.side_effect = start_background
    return svc, runner, SimpleNamespace(task_runner=runner, workflow_service=svc)


async def _post(state, session: str):
    request = _request(state, session=session, json_body={"spec": "__inline__:# task"})
    request["internal_auth"] = True
    return await api_taskrunner_start(request)


async def _settle(svc: WorkflowService, run_id: str) -> None:
    handle = svc.registry.get(run_id)
    assert handle is not None and handle.task is not None
    await asyncio.wait_for(asyncio.shield(handle.task), timeout=_HANG_GUARD_SECS)


@pytest.mark.asyncio
async def test_a_task_run_from_a_worker_is_refused_past_the_depth_cap(tmp_path):
    svc, runner, state = _wired(tmp_path, max_nesting_depth=1)
    root = await svc.start(QUICK, session_key="dashboard:main")
    response = await _post(state, f"wf:{root['run_id']}:0")
    assert response.status == 400
    assert _body(response)["code"] == ERROR_CODE_DEPTH
    runner.start_background.assert_not_awaited()
    await _settle(svc, root["run_id"])


@pytest.mark.asyncio
async def test_a_task_run_from_a_worker_takes_a_fanout_slot(tmp_path):
    svc, runner, state = _wired(tmp_path, max_nested_runs=1)
    root = await svc.start(QUICK, session_key="dashboard:main")
    worker = f"wf:{root['run_id']}:0"
    first = await _post(state, worker)
    assert first.status == 200, _body(first)
    second = await _post(state, worker)
    assert second.status == 400
    assert _body(second)["code"] == ERROR_CODE_FANOUT
    assert runner.start_background.await_count == 1
    await _settle(svc, root["run_id"])


@pytest.mark.asyncio
async def test_a_task_run_outside_any_run_is_not_capped(tmp_path):
    svc, runner, state = _wired(tmp_path, max_nesting_depth=1, max_nested_runs=1)
    for _ in range(2):
        response = await _post(state, "dashboard:main")
        assert response.status == 200, _body(response)
