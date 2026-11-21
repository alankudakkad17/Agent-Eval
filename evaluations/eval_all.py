"""Run all 7 metrics on every golden and save results/latest.json.

Usage:
  uv run python evaluations/eval_all.py                     # all metrics
  uv run python evaluations/eval_all.py "Tool Correctness"  # some metrics
  uv run python evaluations/eval_all.py --retry-errors      # re-score errored checks only
"""
import sys

from common import evaluate_dataset, retry_errors

if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--retry-errors"]:
        retry_errors()
    else:
        evaluate_dataset(args or None)
