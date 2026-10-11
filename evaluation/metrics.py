"""Classification metrics computed only from true labels and predictions.

Undefined values are reported as None instead of being silently set to
0 or 1:
- recall and F1 are None for a class with no samples (support 0);
- precision is None for a class the model never predicted.
Macro F1 averages only classes that have samples, and the report states
how many classes that is. Confidence is summarized separately and is
never used to decide correctness.
"""


def confusion_matrix(y_true, y_pred, num_classes):
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual][predicted] += 1
    return matrix


def classification_metrics(y_true, y_pred, class_names):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    num_classes = len(class_names)
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    total = len(y_true)
    correct = sum(matrix[i][i] for i in range(num_classes))

    per_class = []
    for index, name in enumerate(class_names):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        if not support:
            f1 = None
        elif tp == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_class.append({
            "index": index,
            "food": name,
            "support": support,
            "predicted": predicted,
            "true_positives": tp,
            "false_positives": predicted - tp,
            "false_negatives": support - tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    present = [row for row in per_class if row["support"]]
    macro_f1 = sum(row["f1"] for row in present) / len(present) if present else None
    weighted_f1 = sum(row["f1"] * row["support"] for row in present) / total if total else None

    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total if total else None,
        "macro_f1": macro_f1,
        "macro_f1_classes": len(present),
        "weighted_f1": weighted_f1,
        "classes_without_samples": [row["food"] for row in per_class if not row["support"]],
        "predicted_but_absent": [
            row["food"] for row in per_class if not row["support"] and row["predicted"]
        ],
        "per_class": per_class,
        "confusion_matrix": {"labels": list(class_names), "matrix": matrix},
    }


def confidence_summary(confidences, correct_flags, bins=10):
    """Describes model confidence (0-1) next to observed correctness."""
    if len(confidences) != len(correct_flags):
        raise ValueError("confidences and correct_flags must have the same length.")
    if not confidences:
        return None

    def mean(values):
        return sum(values) / len(values) if values else None

    right = [c for c, ok in zip(confidences, correct_flags) if ok]
    wrong = [c for c, ok in zip(confidences, correct_flags) if not ok]

    table = []
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        members = [
            (c, ok) for c, ok in zip(confidences, correct_flags)
            if low <= c < high or (b == bins - 1 and c == 1.0)
        ]
        table.append({
            "range": [low, high],
            "count": len(members),
            "mean_confidence": mean([c for c, _ in members]),
            "accuracy": mean([1.0 if ok else 0.0 for _, ok in members]),
        })

    return {
        "mean_confidence": mean(confidences),
        "mean_confidence_correct": mean(right),
        "mean_confidence_incorrect": mean(wrong),
        "bins": table,
    }
