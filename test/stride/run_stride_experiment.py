"""Compare EMG classification results for several window strides.

The feature extraction and classifier settings intentionally match ``learn.py``:

* 1,000 Hz sample rate and a 200 ms window
* MAV, RMS, WL, and the same nine two-channel relation features
* DecisionTreeClassifier(max_depth=5, random_state=42)

For every class, windows remain in chronological order. The first 75% are used
for training and the remaining 25% for testing.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np
import sklearn
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

import learn  # noqa: E402  (project module; imported after PROJECT_ROOT is added)


STRIDES_MS = (50, 100, 150, 200)
TRAIN_RATIO = 0.75
CLASS_NAMES = [name for name, _path, _label in learn.CLASSES]
CLASS_LABELS = [label for _name, _path, label in learn.CLASSES]


def make_raw_dataset(path: Path, label: int, step_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Create chronological windows using learn.py's feature extraction."""
    data = learn.load_sensor_data(path)
    features = []
    labels = []

    for start in range(0, len(data) - learn.WINDOW_SIZE + 1, step_size):
        window = data[start : start + learn.WINDOW_SIZE]
        features.append(learn.raw_window_features(window))
        labels.append(label)

    return np.asarray(features), np.asarray(labels)


def split_sequential_by_class(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, dict[str, int]]]:
    """Split each class in time order: first 75% train, final 25% test."""
    train_indices: list[int] = []
    test_indices: list[int] = []
    split_counts: dict[int, dict[str, int]] = {}

    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        split_at = int(len(indices) * TRAIN_RATIO)
        if split_at == 0 or split_at == len(indices):
            raise ValueError(f"label {label}: cannot split {len(indices)} windows")

        train_indices.extend(indices[:split_at])
        test_indices.extend(indices[split_at:])
        split_counts[int(label)] = {
            "total": int(len(indices)),
            "train": int(split_at),
            "test": int(len(indices) - split_at),
        }

    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices],
        split_counts,
    )


def run_one(stride_ms: int) -> dict:
    step_size = learn.SAMPLE_RATE_HZ * stride_ms // 1000
    datasets = []
    source_rows = {}

    for name, relative_path, label in learn.CLASSES:
        source_path = PROJECT_ROOT / relative_path
        source_rows[name] = int(len(learn.load_sensor_data(source_path)))
        raw_X, class_y = make_raw_dataset(source_path, label, step_size)
        datasets.append((raw_X, class_y))

    raw_X = np.vstack([class_X for class_X, _class_y in datasets])
    y = np.concatenate([class_y for _class_X, class_y in datasets])
    X = learn.engineered_features(raw_X)
    X_train, X_test, y_train, y_test, split_counts = split_sequential_by_class(X, y)

    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "stride_ms": stride_ms,
        "update_hz": 1000.0 / stride_ms,
        "step_samples": step_size,
        "overlap_ms": max(0, learn.WINDOW_MS - stride_ms),
        "overlap_percent": max(0.0, (learn.WINDOW_MS - stride_ms) / learn.WINDOW_MS * 100.0),
        "source_rows": source_rows,
        "split_counts": {
            CLASS_NAMES[CLASS_LABELS.index(label)]: counts
            for label, counts in split_counts.items()
        },
        "train_windows": int(len(X_train)),
        "test_windows": int(len(X_test)),
        "feature_count": int(X.shape[1]),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_precision": float(
            precision_score(y_test, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_test, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_test, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0)
        ),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y_test, y_pred, labels=CLASS_LABELS
        ).astype(int).tolist(),
        "tree_depth": int(model.get_depth()),
        "tree_leaves": int(model.get_n_leaves()),
    }


def write_summary_csv(results: list[dict]) -> None:
    with (SCRIPT_DIR / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "stride_ms",
                "update_hz",
                "overlap_ms",
                "overlap_percent",
                "train_windows",
                "test_windows",
                "accuracy_percent",
                "macro_precision_percent",
                "macro_recall_percent",
                "macro_f1_percent",
                "tree_depth",
                "tree_leaves",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result["stride_ms"],
                    f'{result["update_hz"]:.4f}',
                    result["overlap_ms"],
                    f'{result["overlap_percent"]:.1f}',
                    result["train_windows"],
                    result["test_windows"],
                    f'{result["accuracy"] * 100:.4f}',
                    f'{result["macro_precision"] * 100:.4f}',
                    f'{result["macro_recall"] * 100:.4f}',
                    f'{result["macro_f1"] * 100:.4f}',
                    result["tree_depth"],
                    result["tree_leaves"],
                ]
            )


def write_confusion_csv(result: dict) -> None:
    output_path = SCRIPT_DIR / f'confusion_{result["stride_ms"]}ms.csv'
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *CLASS_NAMES])
        for name, row in zip(CLASS_NAMES, result["confusion_matrix"]):
            writer.writerow([name, *row])


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_report(results: list[dict], metadata: dict) -> None:
    best = max(results, key=lambda item: (item["accuracy"], item["macro_f1"], -item["stride_ms"]))
    lines = [
        "# Stride별 학습 결과",
        "",
        "## 실험 방법",
        "",
        f'- 데이터: 프로젝트 루트의 `none/rock/paper/middle/thumb.txt` ({metadata["channel_count"]}채널)',
        f'- 샘플링/창: {learn.SAMPLE_RATE_HZ} Hz, {learn.WINDOW_MS} ms ({learn.WINDOW_SIZE}샘플)',
        "- 특성: `learn.py`와 동일한 MAV/RMS/WL 및 채널 관계 특성 15개",
        "- 모델: `DecisionTreeClassifier(max_depth=5, random_state=42)`",
        "- 분할: 클래스별 시간 순서를 유지하고 앞 75%를 학습, 뒤 25%를 테스트에 사용",
        "- 기존 `learn.py` 흐름과 같이 전체 데이터를 창으로 만든 후 클래스별로 분할",
        "",
        "## 요약",
        "",
        "| Stride | 판정 주기 | 겹침 | 학습 창 | 테스트 창 | 정확도 | Macro precision | Macro recall | Macro F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for result in results:
        lines.append(
            f'| {result["stride_ms"]} ms | {result["update_hz"]:.2f} Hz | '
            f'{result["overlap_ms"]} ms ({result["overlap_percent"]:.0f}%) | '
            f'{result["train_windows"]} | '
            f'{result["test_windows"]} | {percent(result["accuracy"])} | '
            f'{percent(result["macro_precision"])} | {percent(result["macro_recall"])} | '
            f'{percent(result["macro_f1"])} |'
        )

    lines.extend(
        [
            "",
            f'최고 테스트 정확도: **{best["stride_ms"]} ms ({percent(best["accuracy"])})**.',
            "",
            "## 클래스별 F1",
            "",
            "| Stride | " + " | ".join(CLASS_NAMES) + " |",
            "|---:|" + "---:|" * len(CLASS_NAMES),
        ]
    )

    for result in results:
        values = [percent(result["classification_report"][name]["f1-score"]) for name in CLASS_NAMES]
        lines.append(f'| {result["stride_ms"]} ms | ' + " | ".join(values) + " |")

    for result in results:
        lines.extend(
            [
                "",
                f'## {result["stride_ms"]} ms 상세 결과',
                "",
                "창 개수(학습/테스트): "
                + ", ".join(
                    f'{name} {counts["train"]}/{counts["test"]}'
                    for name, counts in result["split_counts"].items()
                ),
                "",
                "혼동행렬(행 = 실제, 열 = 예측):",
                "",
                "| actual \\ predicted | " + " | ".join(CLASS_NAMES) + " |",
                "|---|" + "---:|" * len(CLASS_NAMES),
            ]
        )
        for name, row in zip(CLASS_NAMES, result["confusion_matrix"]):
            lines.append(f'| {name} | ' + " | ".join(str(value) for value in row) + " |")

    (SCRIPT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    # Validate assumptions shared by all comparisons.
    channel_counts = {
        learn.load_sensor_data(PROJECT_ROOT / relative_path).shape[1]
        for _name, relative_path, _label in learn.CLASSES
    }
    if len(channel_counts) != 1:
        raise ValueError(f"source files have inconsistent channel counts: {channel_counts}")
    channel_count = channel_counts.pop()
    if channel_count != 2:
        raise ValueError(f"expected the existing two-channel feature method, got {channel_count} channels")
    if STRIDES_MS[-1] != learn.WINDOW_MS:
        raise ValueError("the non-overlapping stride must equal the window length")

    results = [run_one(stride_ms) for stride_ms in STRIDES_MS]
    metadata = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "sample_rate_hz": learn.SAMPLE_RATE_HZ,
        "window_ms": learn.WINDOW_MS,
        "window_samples": learn.WINDOW_SIZE,
        "train_ratio": TRAIN_RATIO,
        "split_mode": "sequential_by_class_after_windowing",
        "class_order": CLASS_NAMES,
        "channel_count": channel_count,
        "feature_names": learn.feature_names(channel_count),
        "model": "DecisionTreeClassifier(max_depth=5, random_state=42)",
    }

    with (SCRIPT_DIR / "results.json").open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "results": results}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    write_summary_csv(results)
    for result in results:
        write_confusion_csv(result)
    write_report(results, metadata)

    for result in results:
        print(
            f'{result["stride_ms"]:>3} ms: '
            f'accuracy={percent(result["accuracy"])}, '
            f'macro_f1={percent(result["macro_f1"])}, '
            f'train/test={result["train_windows"]}/{result["test_windows"]}'
        )
    print(f"Wrote results under: {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
