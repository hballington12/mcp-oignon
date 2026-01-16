"""Task schema and loader for oignon benchmarks."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Trial:
    """A single trial within a task."""

    id: str
    prompt: str


@dataclass
class Task:
    """A benchmark task with multiple trials."""

    task_type: str
    category: str
    paper: dict[str, Any]  # Ground truth paper metadata
    trials: list[Trial]
    grader: dict[str, Any]
    source_file: str


def load_task(filepath: str | Path) -> Task:
    """Load a task from a YAML file.

    Args:
        filepath: Path to the YAML task file

    Returns:
        Task object
    """
    filepath = Path(filepath)

    with open(filepath) as f:
        data = yaml.safe_load(f)

    trials = [Trial(id=t["id"], prompt=t["prompt"]) for t in data["trials"]]

    return Task(
        task_type=data["task_type"],
        category=data["category"],
        paper=data["paper"],
        trials=trials,
        grader=data["grader"],
        source_file=str(filepath),
    )


def load_tasks_from_directory(dirpath: str | Path) -> list[Task]:
    """Load all tasks from a directory (recursively).

    Args:
        dirpath: Path to directory containing YAML task files

    Returns:
        List of Task objects
    """
    dirpath = Path(dirpath)
    tasks = []

    for root, _, files in os.walk(dirpath):
        for filename in files:
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                filepath = Path(root) / filename
                try:
                    task = load_task(filepath)
                    tasks.append(task)
                except Exception as e:
                    print(f"Warning: Failed to load {filepath}: {e}")

    return tasks


def get_all_trials(tasks: list[Task]) -> list[tuple[Task, Trial]]:
    """Flatten tasks into (task, trial) pairs.

    Args:
        tasks: List of Task objects

    Returns:
        List of (Task, Trial) tuples
    """
    pairs = []
    for task in tasks:
        for trial in task.trials:
            pairs.append((task, trial))
    return pairs


def filter_tasks(
    tasks: list[Task],
    category: str | None = None,
    task_type: str | None = None,
) -> list[Task]:
    """Filter tasks by category or type.

    Args:
        tasks: List of Task objects
        category: Filter by category (e.g., "finding_papers")
        task_type: Filter by task type (e.g., "find_paper")

    Returns:
        Filtered list of tasks
    """
    result = tasks

    if category:
        result = [t for t in result if t.category == category]

    if task_type:
        result = [t for t in result if t.task_type == task_type]

    return result
