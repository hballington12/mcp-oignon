"""Run oignon benchmark evaluations."""

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.graders import GradeResult, grade_task
from benchmarks.harness import TaskResult, reset_graph_store, run_task
from benchmarks.tasks import Task, Trial, get_all_trials, load_tasks_from_directory


async def run_trial(
    task: Task,
    trial: Trial,
    model: str = "claude-haiku-4-5",
    max_turns: int = 10,
) -> tuple[TaskResult, GradeResult]:
    """Run a single trial and grade it.

    Args:
        task: The task containing ground truth
        trial: The specific trial to run
        model: Claude model to use
        max_turns: Maximum agent turns

    Returns:
        Tuple of (TaskResult, GradeResult)
    """
    reset_graph_store()

    result = await run_task(
        task=trial.prompt,
        model=model,
        max_turns=max_turns,
    )

    grade = grade_task(
        result=result,
        grader_config=task.grader,
        task_data={"paper": task.paper},
    )

    return result, grade


async def run_eval(
    tasks_dir: str | Path,
    output_dir: str | Path,
    model: str = "claude-haiku-4-5",
    max_turns: int = 10,
    category: str | None = None,
    limit: int | None = None,
    parallel: int = 5,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run evaluation on all tasks.

    Args:
        tasks_dir: Directory containing task YAML files
        output_dir: Directory to write results
        model: Claude model to use
        max_turns: Maximum agent turns per trial
        category: Filter to specific category
        limit: Limit number of trials to run
        parallel: Number of trials to run concurrently (default 5)
        verbose: Print progress

    Returns:
        Summary statistics
    """
    tasks_dir = Path(tasks_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tasks
    tasks = load_tasks_from_directory(tasks_dir)
    if category:
        tasks = [t for t in tasks if t.category == category]

    all_trials = get_all_trials(tasks)

    if limit:
        all_trials = all_trials[:limit]

    if verbose:
        print(f"Running {len(all_trials)} trials from {len(tasks)} tasks")
        print(f"Model: {model}, Parallel: {parallel}")
        print(f"Output: {output_dir}")
        print("-" * 60)

    # Run trials in parallel with semaphore
    semaphore = asyncio.Semaphore(parallel)
    results = []
    completed = 0

    async def run_single(idx: int, task: Task, trial: Trial) -> dict[str, Any]:
        nonlocal completed
        trial_id = f"{Path(task.source_file).stem}/{trial.id}"

        async with semaphore:
            try:
                task_result, grade_result = await run_trial(
                    task=task,
                    trial=trial,
                    model=model,
                    max_turns=max_turns,
                )

                completed += 1
                status = "PASS" if grade_result.passed else f"FAIL - {grade_result.reason}"
                if verbose:
                    print(f"[{completed}/{len(all_trials)}] {trial_id}: {status}")

                return {
                    "task_file": task.source_file,
                    "trial_id": trial.id,
                    "prompt": trial.prompt,
                    "passed": grade_result.passed,
                    "score": grade_result.score,
                    "reason": grade_result.reason,
                    "details": grade_result.details,
                    "turns": task_result.turns,
                    "tool_calls": [
                        {"tool": tc.tool, "input": tc.input}
                        for tc in task_result.tool_calls
                    ],
                    "final_response": task_result.final_response,
                    "error": task_result.error,
                }

            except Exception as e:
                completed += 1
                if verbose:
                    print(f"[{completed}/{len(all_trials)}] {trial_id}: ERROR - {e}")

                return {
                    "task_file": task.source_file,
                    "trial_id": trial.id,
                    "prompt": trial.prompt,
                    "passed": False,
                    "score": 0.0,
                    "reason": f"Error: {e}",
                    "error": str(e),
                }

    # Launch all trials concurrently (semaphore limits actual parallelism)
    tasks_to_run = [
        run_single(i, task, trial)
        for i, (task, trial) in enumerate(all_trials)
    ]
    results = await asyncio.gather(*tasks_to_run)

    # Count results
    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    # Compute summary
    total = passed + failed
    pass_rate = passed / total if total > 0 else 0.0

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "tasks_dir": str(tasks_dir),
        "category": category,
        "total_trials": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
    }

    if verbose:
        print("-" * 60)
        print(f"Results: {passed}/{total} passed ({pass_rate:.1%})")

    # Write results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"results_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(
            {
                "summary": summary,
                "results": results,
            },
            f,
            indent=2,
        )

    if verbose:
        print(f"Results written to {results_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run oignon benchmarks")
    parser.add_argument(
        "--tasks-dir",
        type=str,
        default="benchmarks/tasks",
        help="Directory containing task YAML files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Directory to write results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-haiku-4-5",
        help="Claude model to use",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum agent turns per trial",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter to specific category",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of trials",
    )
    parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        default=5,
        help="Number of trials to run concurrently (default 5)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print progress",
    )

    args = parser.parse_args()

    asyncio.run(
        run_eval(
            tasks_dir=args.tasks_dir,
            output_dir=args.output_dir,
            model=args.model,
            max_turns=args.max_turns,
            category=args.category,
            limit=args.limit,
            parallel=args.parallel,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()
