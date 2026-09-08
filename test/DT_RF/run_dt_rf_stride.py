"""Compare Decision Tree and Random Forest across four EMG strides.

This experiment reuses the project's existing data loading, 200 ms window,
raw-window feature extraction, and 15-feature engineering from ``learn.py``.
For each stride and each class, chronological windows are split into the first
75% for training and the final 25% for testing.  DT and RF always receive the
same train/test arrays within a stride.

All generated files are written next to this script.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
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

import learn  # noqa: E402  (project module, imported after PROJECT_ROOT is set)


STRIDES_MS = (50, 100, 150, 200)
TRAIN_RATIO = 0.75
RANDOM_SEED = 42
CLASS_NAMES = [name for name, _path, _label in learn.CLASSES]
CLASS_LABELS = [label for _name, _path, label in learn.CLASSES]
LABEL_TO_NAME = dict(zip(CLASS_LABELS, CLASS_NAMES))
DISPLAY_NAMES = {"dt": "DT", "rf": "RF"}


def make_raw_dataset(path: Path, label: int, step_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Build chronological windows using learn.py's raw feature extraction."""
    data = learn.load_sensor_data(path)
    features: list[np.ndarray] = []
    labels: list[int] = []
    for start in range(0, len(data) - learn.WINDOW_SIZE + 1, step_size):
        window = data[start : start + learn.WINDOW_SIZE]
        features.append(learn.raw_window_features(window))
        labels.append(label)
    return np.asarray(features), np.asarray(labels)


def split_sequential_by_class(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, int]]]:
    """Use the first 75% of each class's windows for training, without shuffling."""
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
            "first_test_window": int(split_at),
        }

    train_array = np.asarray(train_indices, dtype=int)
    test_array = np.asarray(test_indices, dtype=int)
    return X[train_array], X[test_array], y[train_array], y[test_array], split_counts


def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, Any]:
    start = time.perf_counter()
    fitted = model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    prediction = fitted.predict(X_test)
    predict_seconds = time.perf_counter() - start
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
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "macro_precision": float(
            precision_score(y_test, prediction, labels=CLASS_LABELS, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_test, prediction, labels=CLASS_LABELS, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_test, prediction, labels=CLASS_LABELS, average="macro", zero_division=0)
        ),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "prediction_microseconds_per_window": float(predict_seconds / len(X_test) * 1_000_000),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=CLASS_LABELS)
        .astype(int)
        .tolist(),
    }
    return result, prediction.astype(int), fitted


def exact_mcnemar(
    dt_prediction: np.ndarray, rf_prediction: np.ndarray, y_test: np.ndarray
) -> dict[str, Any]:
    """Paired, two-sided exact McNemar comparison on the same test windows."""
    dt_correct = dt_prediction == y_test
    rf_correct = rf_prediction == y_test
    dt_only = int(np.sum(dt_correct & ~rf_correct))
    rf_only = int(np.sum(~dt_correct & rf_correct))
    discordant = dt_only + rf_only
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(dt_only, rf_only)
        lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * lower_tail)
    return {
        "both_correct": int(np.sum(dt_correct & rf_correct)),
        "dt_only_correct": dt_only,
        "rf_only_correct": rf_only,
        "both_wrong": int(np.sum(~dt_correct & ~rf_correct)),
        "discordant": discordant,
        "exact_mcnemar_p_value": float(p_value),
    }


def model_details(model_name: str, fitted: Any) -> dict[str, Any]:
    if model_name == "dt":
        return {
            "configuration": "DecisionTreeClassifier(max_depth=5, random_state=42)",
            "depth": int(fitted.get_depth()),
            "leaves": int(fitted.get_n_leaves()),
            "nodes": int(fitted.tree_.node_count),
        }
    return {
        "configuration": "RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42, n_jobs=1)",
        "trees": int(len(fitted.estimators_)),
        "mean_depth": float(np.mean([tree.get_depth() for tree in fitted.estimators_])),
        "mean_leaves": float(np.mean([tree.get_n_leaves() for tree in fitted.estimators_])),
    }


def run_stride(stride_ms: int) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    step_size = learn.SAMPLE_RATE_HZ * stride_ms // 1000
    datasets: list[tuple[np.ndarray, np.ndarray]] = []
    for _class_name, relative_path, label in learn.CLASSES:
        raw_X, class_y = make_raw_dataset(PROJECT_ROOT / relative_path, label, step_size)
        datasets.append((raw_X, class_y))

    raw_X = np.vstack([class_X for class_X, _class_y in datasets])
    y = np.concatenate([class_y for _class_X, class_y in datasets])
    X = learn.engineered_features(raw_X)
    X_train, X_test, y_train, y_test, split_counts = split_sequential_by_class(X, y)

    for class_name, counts in split_counts.items():
        assert counts["train"] == math.floor(counts["total"] * TRAIN_RATIO), class_name
        assert counts["first_test_window"] == counts["train"], class_name
    assert X.shape[1] == len(learn.feature_names(2)) == 15

    models = OrderedDict(
        [
            ("dt", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=5,
                    random_state=RANDOM_SEED,
                    n_jobs=1,
                ),
            ),
        ]
    )
    model_results: dict[str, dict[str, Any]] = OrderedDict()
    predictions: dict[str, np.ndarray] = OrderedDict()
    for model_name, model in models.items():
        result, prediction, fitted = evaluate_model(model, X_train, y_train, X_test, y_test)
        result["model_details"] = model_details(model_name, fitted)
        model_results[model_name] = result
        predictions[model_name] = prediction

    result = {
        "stride_ms": stride_ms,
        "stride_samples": step_size,
        "update_hz": 1000.0 / stride_ms,
        "overlap_ms": max(0, learn.WINDOW_MS - stride_ms),
        "overlap_percent": max(0.0, (learn.WINDOW_MS - stride_ms) / learn.WINDOW_MS * 100.0),
        "total_windows": int(len(X)),
        "train_windows": int(len(X_train)),
        "test_windows": int(len(X_test)),
        "split_counts": split_counts,
        "models": model_results,
        "paired_comparison": exact_mcnemar(predictions["dt"], predictions["rf"], y_test),
    }
    return result, y_test, predictions


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_summary_csv(results: list[dict[str, Any]]) -> None:
    with (SCRIPT_DIR / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "stride_ms",
                "update_hz",
                "overlap_ms",
                "overlap_percent",
                "total_windows",
                "train_windows",
                "test_windows",
                "model",
                "accuracy_percent",
                "balanced_accuracy_percent",
                "macro_precision_percent",
                "macro_recall_percent",
                "macro_f1_percent",
                "fit_seconds",
                "prediction_microseconds_per_window",
            ]
        )
        for stride_result in results:
            for model_name, result in stride_result["models"].items():
                writer.writerow(
                    [
                        stride_result["stride_ms"],
                        f'{stride_result["update_hz"]:.4f}',
                        stride_result["overlap_ms"],
                        f'{stride_result["overlap_percent"]:.1f}',
                        stride_result["total_windows"],
                        stride_result["train_windows"],
                        stride_result["test_windows"],
                        DISPLAY_NAMES[model_name],
                        f'{result["accuracy"] * 100:.4f}',
                        f'{result["balanced_accuracy"] * 100:.4f}',
                        f'{result["macro_precision"] * 100:.4f}',
                        f'{result["macro_recall"] * 100:.4f}',
                        f'{result["macro_f1"] * 100:.4f}',
                        f'{result["fit_seconds"]:.6f}',
                        f'{result["prediction_microseconds_per_window"]:.6f}',
                    ]
                )


def write_paired_csv(results: list[dict[str, Any]]) -> None:
    with (SCRIPT_DIR / "paired_comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "stride_ms",
                "test_windows",
                "both_correct",
                "dt_only_correct",
                "rf_only_correct",
                "both_wrong",
                "exact_mcnemar_p_value",
            ]
        )
        for result in results:
            paired = result["paired_comparison"]
            writer.writerow(
                [
                    result["stride_ms"],
                    result["test_windows"],
                    paired["both_correct"],
                    paired["dt_only_correct"],
                    paired["rf_only_correct"],
                    paired["both_wrong"],
                    f'{paired["exact_mcnemar_p_value"]:.8f}',
                ]
            )


def write_confusion_csv(stride_ms: int, model_name: str, matrix: list[list[int]]) -> None:
    path = SCRIPT_DIR / f"confusion_{model_name}_{stride_ms}ms.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([class_name, *row])


def write_predictions_csv(
    stride_ms: int, y_test: np.ndarray, predictions: dict[str, np.ndarray]
) -> None:
    positions = {label: 0 for label in CLASS_LABELS}
    path = SCRIPT_DIR / f"predictions_{stride_ms}ms.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "test_row",
                "class_test_position",
                "actual_label",
                "actual_class",
                "dt_prediction",
                "dt_class",
                "rf_prediction",
                "rf_class",
                "dt_correct",
                "rf_correct",
            ]
        )
        for row_index, actual in enumerate(y_test):
            actual_int = int(actual)
            dt_value = int(predictions["dt"][row_index])
            rf_value = int(predictions["rf"][row_index])
            writer.writerow(
                [
                    row_index,
                    positions[actual_int],
                    actual_int,
                    LABEL_TO_NAME[actual_int],
                    dt_value,
                    LABEL_TO_NAME[dt_value],
                    rf_value,
                    LABEL_TO_NAME[rf_value],
                    int(dt_value == actual_int),
                    int(rf_value == actual_int),
                ]
            )
            positions[actual_int] += 1


def write_report(payload: dict[str, Any]) -> None:
    results = payload["results"]
    all_combinations = [
        (result["stride_ms"], model_name, model_result)
        for result in results
        for model_name, model_result in result["models"].items()
    ]
    overall_best = max(
        all_combinations,
        key=lambda item: (item[2]["accuracy"], item[2]["macro_f1"], -item[0]),
    )
    best_by_model = {
        model_name: max(
            (
                (result["stride_ms"], result["models"][model_name])
                for result in results
            ),
            key=lambda item: (item[1]["accuracy"], item[1]["macro_f1"], -item[0]),
        )
        for model_name in DISPLAY_NAMES
    }

    lines = [
        "# Stride별 DT·RF 비교 결과",
        "",
        "## 실험 조건",
        "",
        "- 데이터: 프로젝트 루트의 `none.txt`, `rock.txt`, `paper.txt`, `middle.txt`, `thumb.txt`",
        f"- 샘플링: {learn.SAMPLE_RATE_HZ:,} Hz, 창 길이 {learn.WINDOW_MS} ms({learn.WINDOW_SIZE}개 샘플)",
        "- stride: 50/100/150/200 ms",
        "- 특성: 기존 `learn.py`와 동일한 MAV/RMS/WL 및 2채널 관계 특성 15개",
        "- 분할: 각 stride에서 클래스별 시간 순서를 유지해 앞 75% 학습, 뒤 25% 테스트",
        "- DT와 RF는 각 stride 안에서 완전히 동일한 학습·테스트 창 사용",
        "- DT: `DecisionTreeClassifier(max_depth=5, random_state=42)`",
        "- RF: `RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42, n_jobs=1)`",
        "",
        "## 성능 요약",
        "",
        "| stride | overlap | 학습/테스트 창 | DT 정확도 | DT Macro F1 | RF 정확도 | RF Macro F1 | 정확도 차이(RF-DT) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        dt = result["models"]["dt"]
        rf = result["models"]["rf"]
        lines.append(
            f"| {result['stride_ms']} ms | {result['overlap_ms']} ms ({result['overlap_percent']:.0f}%) | "
            f"{result['train_windows']}/{result['test_windows']} | {percent(dt['accuracy'])} | "
            f"{percent(dt['macro_f1'])} | {percent(rf['accuracy'])} | {percent(rf['macro_f1'])} | "
            f"{(rf['accuracy'] - dt['accuracy']) * 100:+.2f}%p |"
        )

    best_stride, best_model, best_result = overall_best
    lines.extend(
        [
            "",
            f"전체 8개 조합 중 최고 결과는 **{best_stride} ms {DISPLAY_NAMES[best_model]}**로, 정확도 {percent(best_result['accuracy'])}, macro F1 {percent(best_result['macro_f1'])}입니다.",
            f"DT의 최고 stride는 **{best_by_model['dt'][0]} ms**(정확도 {percent(best_by_model['dt'][1]['accuracy'])}), RF의 최고 stride는 **{best_by_model['rf'][0]} ms**(정확도 {percent(best_by_model['rf'][1]['accuracy'])})입니다.",
            "",
            "## 동일 테스트 창에서의 직접 비교",
            "",
            "| stride | 둘 다 정답 | DT만 정답 | RF만 정답 | 둘 다 오답 | exact McNemar p |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        paired = result["paired_comparison"]
        lines.append(
            f"| {result['stride_ms']} ms | {paired['both_correct']} | {paired['dt_only_correct']} | "
            f"{paired['rf_only_correct']} | {paired['both_wrong']} | {paired['exact_mcnemar_p_value']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## 클래스별 F1",
            "",
            "| stride | 모델 | " + " | ".join(CLASS_NAMES) + " |",
            "|---:|---|" + "---:|" * len(CLASS_NAMES),
        ]
    )
    for result in results:
        for model_name, model_result in result["models"].items():
            class_values = [
                percent(model_result["classification_report"][name]["f1-score"])
                for name in CLASS_NAMES
            ]
            lines.append(
                f"| {result['stride_ms']} ms | {DISPLAY_NAMES[model_name]} | "
                + " | ".join(class_values)
                + " |"
            )

    lines.extend(
        [
            "",
            "## 혼동행렬",
            "",
            "행은 실제 클래스, 열은 예측 클래스입니다.",
        ]
    )
    for result in results:
        for model_name, model_result in result["models"].items():
            lines.extend(
                [
                    "",
                    f"### {result['stride_ms']} ms · {DISPLAY_NAMES[model_name]}",
                    "",
                    "| actual \\ predicted | " + " | ".join(CLASS_NAMES) + " |",
                    "|---|" + "---:|" * len(CLASS_NAMES),
                ]
            )
            for class_name, row in zip(CLASS_NAMES, model_result["confusion_matrix"]):
                lines.append(f"| {class_name} | " + " | ".join(str(value) for value in row) + " |")

    lines.extend(
        [
            "",
            "## 해석 시 주의사항",
            "",
            "- stride가 짧을수록 같은 원시 기록에서 서로 겹치는 창이 더 많이 생성됩니다. 창 수 증가는 새로운 독립 데이터가 늘어난 것이 아닙니다.",
            "- 기존 방식처럼 창을 만든 뒤 80/20으로 분할했기 때문에, 50/100/150 ms 조건에서는 분할 경계의 마지막 학습 창과 첫 테스트 창이 일부 원시 샘플을 공유할 수 있습니다. 200 ms 조건은 창이 겹치지 않습니다.",
            "- stride마다 테스트 창 개수가 다르므로 정확도의 1개 오차가 차지하는 비율도 다릅니다. 모델 간 직접 비교는 같은 stride 안의 DT/RF 결과가 가장 타당합니다.",
            "- McNemar 값은 겹치는 창들의 독립성이 보장되지 않으므로 참고용 paired 지표로만 해석해야 합니다.",
            "- 한 수집 세션의 고정 holdout 결과입니다. 최종 선정에는 별도 세션이나 사용자 단위 분리 평가가 필요합니다.",
            "",
            "## 생성 파일",
            "",
            "- `results.json`: 설정과 전체 세부 결과",
            "- `summary.csv`: stride·모델별 핵심 지표",
            "- `paired_comparison.csv`: 동일 테스트 창의 DT/RF 직접 비교",
            "- `predictions_*ms.csv`: 테스트 창별 실제값과 두 모델 예측",
            "- `confusion_*_*.csv`: 모델·stride별 혼동행렬",
        ]
    )
    (SCRIPT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    channel_counts = {
        int(learn.load_sensor_data(PROJECT_ROOT / relative_path).shape[1])
        for _name, relative_path, _label in learn.CLASSES
    }
    if channel_counts != {2}:
        raise ValueError(f"expected the existing two-channel data, got {channel_counts}")

    source_rows = {
        class_name: int(len(learn.load_sensor_data(PROJECT_ROOT / relative_path)))
        for class_name, relative_path, _label in learn.CLASSES
    }
    results: list[dict[str, Any]] = []
    for stride_ms in STRIDES_MS:
        result, y_test, predictions = run_stride(stride_ms)
        results.append(result)
        write_predictions_csv(stride_ms, y_test, predictions)
        for model_name, model_result in result["models"].items():
            write_confusion_csv(stride_ms, model_name, model_result["confusion_matrix"])
            print(
                f"{stride_ms:>3} ms {DISPLAY_NAMES[model_name]}: "
                f"accuracy={percent(model_result['accuracy'])}, "
                f"macro_f1={percent(model_result['macro_f1'])}"
            )

    payload = {
        "metadata": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
            "random_seed": RANDOM_SEED,
            "sample_rate_hz": learn.SAMPLE_RATE_HZ,
            "window_ms": learn.WINDOW_MS,
            "window_samples": learn.WINDOW_SIZE,
            "strides_ms": list(STRIDES_MS),
            "train_ratio": TRAIN_RATIO,
            "split_mode": "sequential_by_class_after_windowing",
            "feature_count": 15,
            "feature_names": learn.feature_names(2),
            "class_order": CLASS_NAMES,
            "class_labels": CLASS_LABELS,
            "source_rows": source_rows,
            "models": {
                "dt": "DecisionTreeClassifier(max_depth=5, random_state=42)",
                "rf": "RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42, n_jobs=1)",
            },
        },
        "results": results,
    }
    with (SCRIPT_DIR / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_summary_csv(results)
    write_paired_csv(results)
    write_report(payload)
    print(f"Wrote results under: {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
