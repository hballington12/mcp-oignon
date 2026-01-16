"""Graders for oignon benchmark tasks.

Graders evaluate agent performance by checking transcripts and tool outputs.
"""

import json
import re
from dataclasses import dataclass
from typing import Any

from benchmarks.harness import TaskResult, ToolCall


@dataclass
class GradeResult:
    """Result of grading a task."""

    passed: bool
    score: float  # 0.0 to 1.0
    reason: str
    details: dict[str, Any] | None = None


def paper_id_matches(
    result: TaskResult,
    expected_id: str,
) -> GradeResult:
    """Check if the agent found the correct paper.

    Searches through tool outputs, final response, and transcript for the
    expected OpenAlex ID. The ID can appear with or without the full URL prefix.

    Args:
        result: The TaskResult from running the agent
        expected_id: Expected OpenAlex ID (e.g., "W1970754130")

    Returns:
        GradeResult with pass/fail and details
    """
    # Normalize expected ID - strip URL prefix if present
    if expected_id.startswith("https://openalex.org/"):
        expected_id = expected_id.replace("https://openalex.org/", "")

    # Places to look for the ID
    found_in: list[str] = []

    # Check tool outputs
    for tc in result.tool_calls:
        output = tc.output
        if expected_id in output:
            found_in.append(f"tool:{tc.tool}")
        # Also check for full URL form
        if f"https://openalex.org/{expected_id}" in output:
            found_in.append(f"tool:{tc.tool}")

    # Check final response
    if result.final_response and expected_id in result.final_response:
        found_in.append("final_response")

    # Check transcript (contains raw messages including tool results)
    transcript_str = json.dumps(result.transcript)
    if expected_id in transcript_str:
        found_in.append("transcript")

    passed = len(found_in) > 0

    return GradeResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        reason=f"Found {expected_id} in: {', '.join(found_in)}" if passed else f"Did not find {expected_id}",
        details={
            "expected_id": expected_id,
            "found_in": found_in,
            "tool_calls": [tc.tool for tc in result.tool_calls],
            "turns": result.turns,
        },
    )


def paper_id_in_top_n(
    result: TaskResult,
    expected_id: str,
    n: int = 5,
) -> GradeResult:
    """Check if the correct paper appears in the top N search results.

    More lenient than paper_id_matches - the agent doesn't need to highlight
    the correct paper, just include it in search results.

    Args:
        result: The TaskResult from running the agent
        expected_id: Expected OpenAlex ID
        n: Check top N results (default 5)

    Returns:
        GradeResult with pass/fail and position details
    """
    if expected_id.startswith("https://openalex.org/"):
        expected_id = expected_id.replace("https://openalex.org/", "")

    # Look for search_paper tool calls
    for tc in result.tool_calls:
        if tc.tool == "search_paper":
            try:
                output = json.loads(tc.output)
                if isinstance(output, list):
                    for i, paper in enumerate(output[:n]):
                        paper_id = paper.get("id", "")
                        if expected_id in paper_id:
                            return GradeResult(
                                passed=True,
                                score=1.0,
                                reason=f"Found {expected_id} at position {i + 1}",
                                details={"position": i + 1, "n": n},
                            )
            except (json.JSONDecodeError, TypeError):
                continue

    return GradeResult(
        passed=False,
        score=0.0,
        reason=f"Did not find {expected_id} in top {n} results",
        details={"n": n},
    )


# Registry of grader functions
GRADERS = {
    "paper_id_matches": paper_id_matches,
    "paper_id_in_top_n": paper_id_in_top_n,
}


def grade_task(
    result: TaskResult,
    grader_config: dict[str, Any],
    task_data: dict[str, Any],
) -> GradeResult:
    """Grade a task result using the specified grader.

    Args:
        result: The TaskResult from running the agent
        grader_config: Grader configuration from task YAML
        task_data: Full task data including expected values

    Returns:
        GradeResult
    """
    grader_type = grader_config.get("type")
    check = grader_config.get("check")

    if grader_type != "code":
        raise ValueError(f"Unknown grader type: {grader_type}")

    if check not in GRADERS:
        raise ValueError(f"Unknown grader check: {check}")

    grader_fn = GRADERS[check]

    # Extract expected values based on check type
    if check == "paper_id_matches":
        expected_id = task_data["paper"]["openalex_id"]
        return grader_fn(result, expected_id)

    elif check == "paper_id_in_top_n":
        expected_id = task_data["paper"]["openalex_id"]
        n = grader_config.get("n", 5)
        return grader_fn(result, expected_id, n=n)

    else:
        raise ValueError(f"Unhandled check: {check}")
