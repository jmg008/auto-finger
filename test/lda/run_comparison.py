"""Compare the project's Decision Tree and LDA classifiers.

The input parsing, windowing, and feature extraction intentionally reuse
``learn.py``.  For each class, chronological feature windows are split into
the first 75% for training and the remaining 25% for testing.

All generated artifacts are written next to this script.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import learn  # noqa: E402  (import after adding the project root)


TRAIN_RATIO = 0.75
CLASS_NAMES = [name for name, _path, _label in learn.CLASSES]
CLASS_LABELS = [label for _name, _path, label in learn.CLASSES]
LABEL_TO_NAME = dict(zip(CLASS_LABELS, CLASS_NAMES))


def split_sequential_by_class(
    X: np.ndarray, y: np.ndarray
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, dict[str, int]],
    np.ndarray,
]:
    """Use each class's first 75% of windows for training, without shuffling."""
    train_indices: list[int] = []
    test_indices: list[int] = []
    split_counts: dict[str, dict[str, int]] = {}

    for label in CLASS_LABELS:
        indices = np.flatnonzero(y == label)
        split_at = int(len(indices) * TRAIN_RATIO)
        if split_at == 0 or split_at == len(indices):
            raise ValueError(f"label {label}: cannot split {len(indices)} windows")

        class_train = indices[:split_at]
        class_test = indices[split_at:]
        train_indices.extend(class_train.tolist())
        test_indices.extend(class_test.tolist())
        split_counts[LABEL_TO_NAME[label]] = {
            "total": int(len(indices)),
            "train": int(len(class_train)),
            "test": int(len(class_test)),
            "first_train_window": 0,
            "last_train_window": int(split_at - 1),
            "first_test_window": int(split_at),
            "last_test_window": int(len(indices) - 1),
        }

    train_index_array = np.asarray(train_indices, dtype=int)
    test_index_array = np.asarray(test_indices, dtype=int)
    return (
        X[train_index_array],
        X[test_index_array],
        y[train_index_array],
        y[test_index_array],
        split_counts,
        test_index_array,
    )


def exact_mcnemar_p_value(dt_correct: np.ndarray, lda_correct: np.ndarray) -> tuple[int, int, float]:
    """Return discordant counts and the two-sided exact McNemar p-value."""
    dt_only = int(np.sum(dt_correct & ~lda_correct))
    lda_only = int(np.sum(~dt_correct & lda_correct))
    discordant = dt_only + lda_only
    if discordant == 0:
        return dt_only, lda_only, 1.0

    smaller = min(dt_only, lda_only)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return dt_only, lda_only, min(1.0, 2.0 * tail)


def evaluate_model(model: object, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> tuple[dict, np.ndarray]:
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    report = classification_report(
        y_test,
        prediction,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    result = {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "macro_precision": float(
            precision_score(y_test, prediction, labels=CLASS_LABELS, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_test, prediction, labels=CLASS_LABELS, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_test, prediction, labels=CLASS_LABELS, average="macro", zero_division=0)
        ),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=CLASS_LABELS)
        .astype(int)
        .tolist(),
    }
    return result, prediction.astype(int)


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_summary_csv(results: dict[str, dict]) -> None:
    with (SCRIPT_DIR / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "accuracy_percent", "macro_precision_percent", "macro_recall_percent", "macro_f1_percent"])
        for model_name, result in results.items():
            writer.writerow(
                [
                    model_name,
                    f'{result["accuracy"] * 100:.4f}',
                    f'{result["macro_precision"] * 100:.4f}',
                    f'{result["macro_recall"] * 100:.4f}',
                    f'{result["macro_f1"] * 100:.4f}',
                ]
            )


def write_confusion_csv(model_name: str, matrix: list[list[int]]) -> None:
    with (SCRIPT_DIR / f"confusion_{model_name}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([class_name, *row])


def write_predictions_csv(y_test: np.ndarray, predictions: dict[str, np.ndarray]) -> None:
    per_class_positions = {label: 0 for label in CLASS_LABELS}
    with (SCRIPT_DIR / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "test_row",
                "class_test_position",
                "actual_label",
                "actual_class",
                "dt_prediction",
                "dt_class",
                "lda_prediction",
                "lda_class",
                "dt_correct",
                "lda_correct",
            ]
        )
        for row_index, (actual, dt_prediction, lda_prediction) in enumerate(
            zip(y_test, predictions["dt"], predictions["lda"])
        ):
            actual_int = int(actual)
            writer.writerow(
                [
                    row_index,
                    per_class_positions[actual_int],
                    actual_int,
                    LABEL_TO_NAME[actual_int],
                    int(dt_prediction),
                    LABEL_TO_NAME[int(dt_prediction)],
                    int(lda_prediction),
                    LABEL_TO_NAME[int(lda_prediction)],
                    int(dt_prediction == actual),
                    int(lda_prediction == actual),
                ]
            )
            per_class_positions[actual_int] += 1


def markdown_confusion(model_name: str, matrix: list[list[int]]) -> list[str]:
    lines = [
        f"### {model_name.upper()} 혼동 행렬",
        "",
        "행은 실제 클래스, 열은 예측 클래스입니다.",
        "",
        "| actual \\ predicted | " + " | ".join(CLASS_NAMES) + " |",
        "|---|" + "---:|" * len(CLASS_NAMES),
    ]
    for class_name, row in zip(CLASS_NAMES, matrix):
        lines.append(f"| {class_name} | " + " | ".join(str(value) for value in row) + " |")
    return lines


def write_report(payload: dict) -> None:
    results = payload["results"]
    comparison = payload["paired_comparison"]
    winner = max(results, key=lambda name: (results[name]["accuracy"], results[name]["macro_f1"]))
    loser = "lda" if winner == "dt" else "dt"
    accuracy_gap = results[winner]["accuracy"] - results[loser]["accuracy"]

    lines = [
        "# DT와 LDA 비교 결과",
        "",
        "## 실험 조건",
        "",
        "- 데이터: 프로젝트 루트의 `none.txt`, `rock.txt`, `paper.txt`, `middle.txt`, `thumb.txt`",
        f'- 표본화/창: {learn.SAMPLE_RATE_HZ:,} Hz, {learn.WINDOW_MS} ms 창({learn.WINDOW_SIZE}개 샘플), {learn.STEP_MS} ms stride',
        "- 특성: 기존 `learn.py`와 같은 MAV/RMS/WL 및 채널 관계 특성 15개",
        "- 분할: 각 클래스의 창 순서를 유지해 앞 75%를 학습, 뒤 25%를 테스트에 사용",
        "- DT: `DecisionTreeClassifier(max_depth=5, random_state=42)`",
        "- LDA: `LinearDiscriminantAnalysis(solver=\"svd\")`",
        "- 두 모델은 완전히 동일한 학습/테스트 행과 특성을 사용",
        "",
        "## 데이터 분할",
        "",
        "| class | 전체 창 | 학습(앞 75%) | 테스트(뒤 25%) |",
        "|---|---:|---:|---:|",
    ]
    for class_name, counts in payload["metadata"]["split_counts"].items():
        lines.append(f'| {class_name} | {counts["total"]} | {counts["train"]} | {counts["test"]} |')

    lines.extend(
        [
            f'| **합계** | **{payload["metadata"]["total_windows"]}** | **{payload["metadata"]["train_windows"]}** | **{payload["metadata"]["test_windows"]}** |',
            "",
            "## 성능 요약",
            "",
            "| 모델 | 정확도 | Macro precision | Macro recall | Macro F1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model_name, result in results.items():
        lines.append(
            f'| {model_name.upper()} | {percent(result["accuracy"])} | '
            f'{percent(result["macro_precision"])} | {percent(result["macro_recall"])} | '
            f'{percent(result["macro_f1"])} |'
        )

    lines.extend(
        [
            "",
            f'이 홀드아웃에서는 **{winner.upper()}가 정확도 기준 {accuracy_gap * 100:.2f}%p 높았습니다.**',
            "",
            "## 클래스별 F1",
            "",
            "| class | DT | LDA | 차이(LDA-DT) |",
            "|---|---:|---:|---:|",
        ]
    )
    for class_name in CLASS_NAMES:
        dt_f1 = results["dt"]["classification_report"][class_name]["f1-score"]
        lda_f1 = results["lda"]["classification_report"][class_name]["f1-score"]
        lines.append(f"| {class_name} | {percent(dt_f1)} | {percent(lda_f1)} | {(lda_f1 - dt_f1) * 100:+.2f}%p |")

    lines.extend(
        [
            "",
            "## 동일 테스트 행에서의 직접 비교",
            "",
            f'- 둘 다 정답: {comparison["both_correct"]}개',
            f'- DT만 정답: {comparison["dt_only_correct"]}개',
            f'- LDA만 정답: {comparison["lda_only_correct"]}개',
            f'- 둘 다 오답: {comparison["both_wrong"]}개',
            f'- exact McNemar p-value: {comparison["exact_mcnemar_p_value"]:.6f}',
            "",
            "McNemar 검정은 두 모델이 서로 다르게 맞힌 테스트 행만 비교합니다. 다만 200 ms 창을 100 ms 간격으로 만들었으므로 인접 테스트 창끼리 겹쳐 독립 표본 가정이 약합니다. 따라서 이 p-value는 보조적인 paired 지표로만 해석해야 합니다.",
            "",
        ]
    )
    lines.extend(markdown_confusion("dt", results["dt"]["confusion_matrix"]))
    lines.append("")
    lines.extend(markdown_confusion("lda", results["lda"]["confusion_matrix"]))
    lines.extend(
        [
            "",
            "## 해석과 주의점",
            "",
            "- DT는 비선형 임계값과 클래스별 국소 패턴을 직접 나눌 수 있고, 현재 프로젝트의 C 헤더 생성 경로와 바로 연결됩니다.",
            "- LDA는 클래스가 공통 공분산을 가진 선형 경계로 분리된다고 가정합니다. 모델은 단순하지만 현재 특성의 실제 분포가 이 가정과 다르면 성능이 낮아질 수 있습니다.",
            "- 단일 sequential holdout 결과이므로 다른 수집 세션이나 사용자에 대한 일반화 성능을 뜻하지는 않습니다.",
            "- 200 ms 창을 100 ms 간격으로 만들기 때문에 인접 창끼리 100 ms를 공유하며, 기존 방식대로 창을 먼저 만든 뒤 분할해 마지막 학습 창과 첫 테스트 창도 겹칩니다. 완전히 분리된 평가가 필요하면 원시 샘플을 먼저 3:1로 나눈 뒤 각각 창을 만드는 추가 실험이 필요합니다.",
        ]
    )
    (SCRIPT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    datasets = []
    source_rows: dict[str, int] = {}
    channel_counts: set[int] = set()

    for class_name, relative_path, label in learn.CLASSES:
        source_path = PROJECT_ROOT / relative_path
        sensor_data = learn.load_sensor_data(source_path)
        source_rows[class_name] = int(len(sensor_data))
        channel_counts.add(int(sensor_data.shape[1]))
        raw_X, class_y = learn.make_raw_dataset(source_path, label)
        datasets.append((raw_X, class_y))

    if channel_counts != {2}:
        raise ValueError(f"expected the existing two-channel method, got {channel_counts}")

    raw_X = np.vstack([class_X for class_X, _class_y in datasets])
    y = np.concatenate([class_y for _class_X, class_y in datasets])
    X = learn.engineered_features(raw_X)
    X_train, X_test, y_train, y_test, split_counts, test_indices = split_sequential_by_class(X, y)

    # Guard the central requirement: every class must be split at floor(75%).
    for class_name, counts in split_counts.items():
        assert counts["train"] == int(counts["total"] * TRAIN_RATIO), class_name
        assert counts["first_test_window"] == counts["train"], class_name
    assert X.shape[1] == len(learn.feature_names(2)) == 15

    models = OrderedDict(
        [
            ("dt", DecisionTreeClassifier(max_depth=5, random_state=42)),
            ("lda", LinearDiscriminantAnalysis(solver="svd")),
        ]
    )
    results: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}
    for model_name, model in models.items():
        results[model_name], predictions[model_name] = evaluate_model(
            model, X_train, y_train, X_test, y_test
        )

    dt_model = models["dt"]
    results["dt"]["model_details"] = {
        "parameters": "DecisionTreeClassifier(max_depth=5, random_state=42)",
        "depth": int(dt_model.get_depth()),
        "leaves": int(dt_model.get_n_leaves()),
        "nodes": int(dt_model.tree_.node_count),
    }
    lda_model = models["lda"]
    results["lda"]["model_details"] = {
        "parameters": 'LinearDiscriminantAnalysis(solver="svd")',
        "classes": int(len(lda_model.classes_)),
        "discriminant_components": int(lda_model.scalings_.shape[1]),
        "explained_variance_ratio": [float(value) for value in lda_model.explained_variance_ratio_],
    }

    dt_correct = predictions["dt"] == y_test
    lda_correct = predictions["lda"] == y_test
    dt_only, lda_only, p_value = exact_mcnemar_p_value(dt_correct, lda_correct)
    paired_comparison = {
        "both_correct": int(np.sum(dt_correct & lda_correct)),
        "dt_only_correct": dt_only,
        "lda_only_correct": lda_only,
        "both_wrong": int(np.sum(~dt_correct & ~lda_correct)),
        "exact_mcnemar_p_value": float(p_value),
    }

    payload = {
        "metadata": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
            "sample_rate_hz": learn.SAMPLE_RATE_HZ,
            "window_ms": learn.WINDOW_MS,
            "window_samples": learn.WINDOW_SIZE,
            "stride_ms": learn.STEP_MS,
            "stride_samples": learn.STEP_SIZE,
            "train_ratio": TRAIN_RATIO,
            "split_mode": "sequential_by_class_after_windowing",
            "class_order": CLASS_NAMES,
            "class_labels": CLASS_LABELS,
            "source_rows": source_rows,
            "split_counts": split_counts,
            "feature_count": int(X.shape[1]),
            "feature_names": learn.feature_names(2),
            "total_windows": int(len(X)),
            "train_windows": int(len(X_train)),
            "test_windows": int(len(X_test)),
            "test_global_indices": test_indices.tolist(),
        },
        "results": results,
        "paired_comparison": paired_comparison,
    }

    with (SCRIPT_DIR / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_summary_csv(results)
    for model_name, result in results.items():
        write_confusion_csv(model_name, result["confusion_matrix"])
    write_predictions_csv(y_test, predictions)
    write_report(payload)

    for model_name, result in results.items():
        print(
            f"{model_name.upper()}: accuracy={percent(result['accuracy'])}, "
            f"macro_f1={percent(result['macro_f1'])}"
        )
    print(
        f"paired: dt_only={dt_only}, lda_only={lda_only}, "
        f"exact_mcnemar_p={p_value:.6f}"
    )
    print(f"train/test windows: {len(X_train)}/{len(X_test)}")
    print(f"Wrote results under: {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
