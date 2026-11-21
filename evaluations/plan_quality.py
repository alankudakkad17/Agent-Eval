"""Plan Quality only (results are printed, not saved)."""
from common import evaluate_dataset

if __name__ == "__main__":
    evaluate_dataset(["Plan Quality"], save=False)
