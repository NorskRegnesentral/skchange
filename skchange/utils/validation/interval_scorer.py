"""Validation functions for interval scorers."""

from ...base import BaseIntervalScorer
from .enums import EvaluationType


def check_interval_scorer(
    scorer: BaseIntervalScorer,
    arg_name: str,
    caller_name: str,
    required_tasks: list | None = None,
    allow_penalised: bool = True,
    require_evaluation_type: str | None = None,
) -> None:
    """Check if the given scorer is a valid interval scorer."""
    if not isinstance(scorer, BaseIntervalScorer):
        raise ValueError(
            f"`{arg_name}` must be a BaseIntervalScorer. " f"Got {type(scorer)}."
        )
    task = scorer.get_tag("task")
    if task not in required_tasks:
        _required_tasks = [f'"{task}"' for task in required_tasks]
        tasks_str = (
            ", ".join(_required_tasks[:-1]) + " or " + _required_tasks[-1]
            if len(_required_tasks) > 1
            else _required_tasks[0]
        )
        raise ValueError(
            f"{caller_name} requires `{arg_name}` to have task {tasks_str}"
            f" ({arg_name}.get_tag('task') in {required_tasks}). "
            f'Got {scorer.__class__.__name__}, which has task "{task}".'
        )
    if not allow_penalised and scorer.is_penalised_score:
        raise ValueError(f"`{arg_name}` cannot be a penalised score.")

    if (
        (
            require_evaluation_type == "univariate"
            and scorer.evaluation_type != EvaluationType.UNIVARIATE
        )
        or (
            require_evaluation_type == "multivariate"
            and scorer.evaluation_type != EvaluationType.MULTIVARIATE
        )
        or (
            require_evaluation_type == "conditional"
            and scorer.evaluation_type != EvaluationType.CONDITIONAL
        )
    ):
        raise ValueError(
            f"`{caller_name}` requires `{arg_name}` to have {require_evaluation_type} "
            f"{arg_name}.evaluation_type. Got {scorer.evaluation_type}."
        )
