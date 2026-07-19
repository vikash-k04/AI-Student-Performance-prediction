"""Train a final-grade regression model from the supplied student dataset."""
from csv import DictReader
from pathlib import Path
import json
import math

PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT / "dataset" / "student_performance_full.csv"
MODEL_PATH = PROJECT / "models" / "final_grade_model.json"
METRICS_PATH = PROJECT / "models" / "metrics.json"
NUMERIC = ["age", "study_hours", "attendance", "sleep_hours", "previous_grade", "assignments_completed", "practice_tests_taken", "group_study_hours", "motivation_level", "mental_health_score", "screen_time", "social_media_hours","time_management_score",
"notes_quality_score"]
CATEGORICAL = ["gender", "internet_access"]
TARGET = "final_grade"
# Use every row in the supplied dataset. The 80/20 split below is applied
# after loading, so 300,000 rows become 240,000 training and 60,000 test rows.
MAX_ROWS = None


def solve(matrix, vector):
    values = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(len(values)):
        pivot = max(range(column, len(values)), key=lambda row: abs(values[row][column]))
        values[column], values[pivot] = values[pivot], values[column]
        scale = values[column][column]
        if abs(scale) < 1e-10: raise ValueError("Training data lacks sufficient variation.")
        values[column] = [item / scale for item in values[column]]
        for row in range(len(values)):
            if row != column:
                factor = values[row][column]
                values[row] = [a - factor * b for a, b in zip(values[row], values[column])]
    return [row[-1] for row in values]


def vector(row, feature_names):
    result = [float(row[name]) for name in NUMERIC]
    for name in feature_names[len(NUMERIC):]:
        column, expected = name.split("=", 1)
        result.append(1.0 if row[column] == expected else 0.0)
    return result


def main():
    with DATASET.open(newline="", encoding="utf-8") as file:
        rows = []
        for row in DictReader(file):
            rows.append(row)
            if MAX_ROWS is not None and len(rows) == MAX_ROWS: break
    category_values = {column: sorted({row[column] for row in rows}) for column in CATEGORICAL}
    feature_names = NUMERIC[:] + [f"{column}={value}" for column in CATEGORICAL for value in category_values[column][1:]]
    train_rows = [row for index, row in enumerate(rows) if index % 5]
    test_rows = [row for index, row in enumerate(rows) if not index % 5]
    width = len(feature_names) + 1
    xtx, xty = [[0.0] * width for _ in range(width)], [0.0] * width
    for row in train_rows:
        values, target = [1.0] + vector(row, feature_names), float(row[TARGET])
        for i, left in enumerate(values):
            xty[i] += left * target
            matrix_row = xtx[i]
            for j, right in enumerate(values): matrix_row[j] += left * right
    for index in range(1, width): xtx[index][index] += 1.0
    coefficients = solve(xtx, xty)
    actual = [float(row[TARGET]) for row in test_rows]
    predicted = [min(100.0, max(0.0, sum(a*b for a, b in zip(coefficients, [1.0] + vector(row, feature_names))))) for row in test_rows]
    mae = sum(abs(a-p) for a, p in zip(actual, predicted)) / len(actual)
    rmse = math.sqrt(sum((a-p)**2 for a, p in zip(actual, predicted)) / len(actual))
    mean = sum(actual) / len(actual)
    r2 = 1 - sum((a-p)**2 for a, p in zip(actual, predicted)) / sum((a-mean)**2 for a in actual)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    MODEL_PATH.write_text(json.dumps({"feature_names": feature_names, "coefficients": coefficients, "category_values": category_values}, indent=2), encoding="utf-8")
    metrics = {"r2": round(r2, 4), "mae": round(mae, 2), "rmse": round(rmse, 2), "training_rows": len(train_rows), "test_rows": len(test_rows)}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__": main()
