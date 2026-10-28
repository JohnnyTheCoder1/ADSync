"""Season workers never outlive their run, and locks are only reclaimed from processes that are gone."""

import json
import os
import socket
import subprocess
import sys
import time

import psutil
import pytest

from adsync.batch import runner

UNUSED_PID = 4194300  # above any PID Windows or Linux hands out in practice


def _stale_owner(**extra):
    return {"token": "stale", "pid": UNUSED_PID, "host": socket.gethostname(),
            "started": "2000-01-01T00:00:00+00:00", **extra}


@pytest.mark.skipif(os.name != "nt", reason="job objects are Windows-specific")
def test_workers_die_when_their_parent_is_killed():
    parent_code = (
        "import subprocess, sys, time\n"
        "from adsync.batch.runner import _kill_with_parent\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "_kill_with_parent(child)\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    parent = subprocess.Popen([sys.executable, "-c", parent_code], stdout=subprocess.PIPE, text=True)
    child = psutil.Process(int(parent.stdout.readline()))
    try:
        parent.kill()  # like closing the console window or ending the task
        parent.wait()
        child.wait(timeout=10)
    finally:
        if child.is_running():
            child.kill()


def test_lock_whose_worker_is_still_running_is_not_reclaimed(tmp_path):
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lock = tmp_path / "output.lock"
        created = psutil.Process(worker.pid).create_time()
        lock.write_text(json.dumps(_stale_owner(worker_pid=worker.pid, worker_created=created)), encoding="utf-8")
        with pytest.raises(runner.BatchStateError):
            with runner._exclusive_lock(lock, "Output test"):
                pass
    finally:
        worker.kill()
        worker.wait()


def test_reclaim_never_takes_a_lock_that_another_run_just_wrote(tmp_path, monkeypatch):
    lock = tmp_path / "output.lock"
    lock.write_text(json.dumps(_stale_owner()), encoding="utf-8")
    decide = runner._owner_has_exited

    def replaced_while_deciding(owner):
        lock.write_text(json.dumps({**_stale_owner(), "token": "fresh", "pid": os.getpid()}), encoding="utf-8")
        return decide(owner)

    monkeypatch.setattr(runner, "_owner_has_exited", replaced_while_deciding)
    assert runner._reclaim_if_abandoned(lock) is False
    assert json.loads(lock.read_text(encoding="utf-8"))["token"] == "fresh"
