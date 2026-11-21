"""Task Completion only (results are printed, not saved)."""
from common import evaluate_dataset

if __name__ == "__main__":
    evaluate_dataset(["Task Completion"], save=False)
