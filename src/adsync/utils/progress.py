"""Append-only progress events for terminals, screen readers and log files."""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger("adsync")


@dataclass
class _Task:
    description: str
    total: int
    completed: int = 0
    last_percent: int = -10


class Progress:
    """Small task tracker that emits a line at each ten-percent milestone."""

    def __init__(self) -> None:
        self.tasks: list[_Task] = []

    def __enter__(self) -> Progress:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def add_task(self, description: str, *, total: int) -> int:
        task_id = len(self.tasks)
        self.tasks.append(_Task(description, total))
        return task_id

    def update(self, task_id: int, *, completed: int | None = None,
               total: int | None = None) -> None:
        task = self.tasks[task_id]
        if total is not None:
            task.total = total
        if completed is not None:
            task.completed = completed
        percent = min(100, int(100 * task.completed / max(1, task.total)))
        if percent >= task.last_percent + 10 or (percent == 100 and task.last_percent != 100):
            log.info("%s: %d%% (%d/%d)", task.description, percent, task.completed, task.total)
            task.last_percent = percent

    def advance(self, task_id: int, advance: int = 1) -> None:
        self.update(task_id, completed=self.tasks[task_id].completed + advance)
