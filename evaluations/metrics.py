"""Metric definitions: name -> factory returning a fresh DeepEval metric."""
from deepeval.metrics import (
    ArgumentCorrectnessMetric,
    GEval,
    PlanAdherenceMetric,
    PlanQualityMetric,
    StepEfficiencyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)
from deepeval.test_case import SingleTurnParams, ToolCallParams

from model import eval_model

SUCCESSFUL_TASK_DESCRIPTION = """
If the username is missing, ask the user to provide it.
For a refund request, identify the customer and the requested purchases, and
check that each purchase is no more than 30 days old and is not in the future.
Refund only the items requested by the user. If the customer or requested item
cannot be found, do not issue a refund ticket and explain why.
Once the necessary information is available, tell the user whether a refund
ticket was issued and its amount, or explain why no refund was issued.
For an information-only request, answer the question without issuing a ticket.
"""

# Sequential calls keep us inside Groq's free-tier rate limits.
COMMON = dict(model=eval_model, async_mode=False)


def task_completion():
    return TaskCompletionMetric(
        threshold=0.8, task=SUCCESSFUL_TASK_DESCRIPTION, include_reason=True, **COMMON
    )


def step_efficiency():
    return StepEfficiencyMetric(threshold=0.8, include_reason=True, **COMMON)


def plan_quality():
    return PlanQualityMetric(threshold=0.8, include_reason=True, **COMMON)


def plan_adherence():
    return PlanAdherenceMetric(threshold=0.8, include_reason=True, **COMMON)


def tool_correctness():
    return ToolCorrectnessMetric(
        threshold=0.8,
        include_reason=True,
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        **COMMON,
    )


def argument_correctness():
    return ArgumentCorrectnessMetric(threshold=0.8, include_reason=True, **COMMON)


def answer_correctness():
    return GEval(
        name="Answer Correctness",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        evaluation_steps=[
            "Check that the actual output and expected output have the same "
            "logical conclusion about the user's request.",
        ],
        threshold=0.8,
        **COMMON,
    )


METRICS = {
    "Task Completion": task_completion,
    "Step Efficiency": step_efficiency,
    "Plan Quality": plan_quality,
    "Plan Adherence": plan_adherence,
    "Tool Correctness": tool_correctness,
    "Argument Correctness": argument_correctness,
    "Answer Correctness": answer_correctness,
}
TRACE_METRICS = ["Task Completion", "Step Efficiency", "Plan Quality", "Plan Adherence"]
