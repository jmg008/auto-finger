"""Diagnose Decision Tree overfitting while varying maximum depth.

The experiment uses the project's default 100 ms stride, 200 ms windows,
15 engineered features, and chronological 75/25 per-class split.  Alongside
the existing split, a purged split removes the last training window of each
class so that it shares no raw samples with the first test window.

Depth selection uses a second purged chronological split made only inside the
primary training partition.  Final-test curves for every depth are diagnostic;
the recommended depth is selected only from the internal validation results.

All generated files are written next to this script.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import learn  # noqa: E402
import run_dt_rf_stride as base  # noqa: E402


STRIDE_MS = 100
STEP_SIZE = learn.SAMPLE_RATE_HZ * STRIDE_MS // 1000
TRAIN_RATIO = 0.75
RANDOM_SEED = 42
DEPTHS: tuple[int | None, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, None)
PURGE_WINDOWS = max(0, math.ceil(learn.WINDOW_SIZE / STEP_SIZE) - 1)


def depth_name(depth: int | None) -> str:
    return "None" if depth is None else str(depth)


def split_sequential(
    X: np.ndarray,
    y: np.ndarray,
    *,
    purge: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, int]]]:
    """Split per class and optionally purge overlapping train-boundary windows."""
    train_indices: list[int] = []
    test_indices: list[int] = []
    counts: dict[str, dict[str, int]] = {}
    purge_count = PURGE_WINDOWS if purge else 0

    for label in base.CLASS_LABELS:
        indices = np.flatnonzero(y == label)
        split_at = int(len(indices) * TRAIN_RATIO)
        class_train = indices[:split_at]
        class_test = indices[split_at:]
        if purge_count:
            if len(class_train) <= purge_count:
                raise ValueError(f"label {label}: not enough train windows to purge")
            class_train = class_train[:-purge_count]
        if len(class_train) == 0 or len(class_test) == 0:
            raise ValueError(f"label {label}: empty split")
        train_indices.extend(class_train.tolist())
        test_indices.extend(class_test.tolist())
        counts[base.LABEL_TO_NAME[label]] = {
            "total": int(len(indices)),
            "train_before_purge": int(split_at),
            "purged": int(purge_count),
            "train": int(len(class_train)),
            "test": int(len(class_test)),
            "first_test_window": int(split_at),
        }

    train_array = np.asarray(train_indices, dtype=int)
    test_array = np.asarray(test_indices, dtype=int)
    return X[train_array], X[test_array], y[train_array], y[test_array], counts


def build_dataset() -> tuple[np.ndarray, np.ndarray]:
    datasets: list[tuple[np.ndarray, np.ndarray]] = []
    for _class_name, relative_path, label in learn.CLASSES:
        raw_X, class_y = base.make_raw_dataset(
            PROJECT_ROOT / relative_path,
            label,
            STEP_SIZE,
        )
        datasets.append((raw_X, class_y))
    raw_X = np.vstack([class_X for class_X, _class_y in datasets])
    y = np.concatenate([class_y for _class_X, class_y in datasets])
    return learn.engineered_features(raw_X), y


def score_predictions(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(
            f1_score(
                y_true,
                prediction,
                labels=base.CLASS_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
    }


def fit_and_evaluate(
    depth: int | None,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)
    train_metrics = score_predictions(y_train, train_prediction)
    test_metrics = score_predictions(y_test, test_prediction)
    report = classification_report(
        y_test,
        test_prediction,
        labels=base.CLASS_LABELS,
        target_names=base.CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return (
        {
            "configured_max_depth": depth,
            "actual_depth": int(model.get_depth()),
            "leaves": int(model.get_n_leaves()),
            "nodes": int(model.tree_.node_count),
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "accuracy_gap": train_metrics["accuracy"] - test_metrics["accuracy"],
            "macro_f1_gap": train_metrics["macro_f1"] - test_metrics["macro_f1"],
            "classification_report": report,
            "confusion_matrix": confusion_matrix(
                y_test, test_prediction, labels=base.CLASS_LABELS
            )
            .astype(int)
            .tolist(),
        },
        test_prediction.astype(int),
    )


def validation_evaluate(
    depth: int | None,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
) -> dict[str, Any]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_SEED)
    model.fit(X_fit, y_fit)
    fit_metrics = score_predictions(y_fit, model.predict(X_fit))
    validation_metrics = score_predictions(y_validation, model.predict(X_validation))
    return {
        "configured_max_depth": depth,
        "actual_depth": int(model.get_depth()),
        "leaves": int(model.get_n_leaves()),
        "fit_accuracy": fit_metrics["accuracy"],
        "fit_macro_f1": fit_metrics["macro_f1"],
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_macro_f1": validation_metrics["macro_f1"],
        "accuracy_gap": fit_metrics["accuracy"] - validation_metrics["accuracy"],
        "macro_f1_gap": fit_metrics["macro_f1"] - validation_metrics["macro_f1"],
    }


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_summary_csv(results: list[dict[str, Any]]) -> None:
    fields = [
        "configured_max_depth",
        "validation_actual_depth",
        "validation_leaves",
        "internal_fit_accuracy_percent",
        "internal_validation_accuracy_percent",
        "internal_accuracy_gap_percent_point",
        "internal_fit_macro_f1_percent",
        "internal_validation_macro_f1_percent",
        "internal_macro_f1_gap_percent_point",
        "purged_actual_depth",
        "purged_leaves",
        "purged_nodes",
        "purged_train_accuracy_percent",
        "purged_test_accuracy_percent",
        "purged_accuracy_gap_percent_point",
        "purged_train_macro_f1_percent",
        "purged_test_macro_f1_percent",
        "purged_macro_f1_gap_percent_point",
        "standard_test_accuracy_percent",
        "standard_test_macro_f1_percent",
    ]
    with (SCRIPT_DIR / "dt_depth_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            validation = result["internal_validation"]
            purged = result["purged_final"]
            standard = result["standard_final"]
            writer.writerow(
                {
                    "configured_max_depth": depth_name(result["configured_max_depth"]),
                    "validation_actual_depth": validation["actual_depth"],
                    "validation_leaves": validation["leaves"],
                    "internal_fit_accuracy_percent": f'{validation["fit_accuracy"] * 100:.4f}',
                    "internal_validation_accuracy_percent": f'{validation["validation_accuracy"] * 100:.4f}',
                    "internal_accuracy_gap_percent_point": f'{validation["accuracy_gap"] * 100:.4f}',
                    "internal_fit_macro_f1_percent": f'{validation["fit_macro_f1"] * 100:.4f}',
                    "internal_validation_macro_f1_percent": f'{validation["validation_macro_f1"] * 100:.4f}',
                    "internal_macro_f1_gap_percent_point": f'{validation["macro_f1_gap"] * 100:.4f}',
                    "purged_actual_depth": purged["actual_depth"],
                    "purged_leaves": purged["leaves"],
                    "purged_nodes": purged["nodes"],
                    "purged_train_accuracy_percent": f'{purged["train_accuracy"] * 100:.4f}',
                    "purged_test_accuracy_percent": f'{purged["test_accuracy"] * 100:.4f}',
                    "purged_accuracy_gap_percent_point": f'{purged["accuracy_gap"] * 100:.4f}',
                    "purged_train_macro_f1_percent": f'{purged["train_macro_f1"] * 100:.4f}',
                    "purged_test_macro_f1_percent": f'{purged["test_macro_f1"] * 100:.4f}',
                    "purged_macro_f1_gap_percent_point": f'{purged["macro_f1_gap"] * 100:.4f}',
                    "standard_test_accuracy_percent": f'{standard["test_accuracy"] * 100:.4f}',
                    "standard_test_macro_f1_percent": f'{standard["test_macro_f1"] * 100:.4f}',
                }
            )


def write_predictions(y_test: np.ndarray, predictions: dict[str, np.ndarray]) -> None:
    positions = {label: 0 for label in base.CLASS_LABELS}
    depth_labels = [depth_name(depth) for depth in DEPTHS]
    with (SCRIPT_DIR / "dt_depth_predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["test_row", "class_test_position", "actual_label", "actual_class"]
            + [f"depth_{label}_prediction" for label in depth_labels]
            + [f"depth_{label}_correct" for label in depth_labels]
        )
        for row_index, actual in enumerate(y_test):
            actual_int = int(actual)
            predicted = [int(predictions[label][row_index]) for label in depth_labels]
            writer.writerow(
                [row_index, positions[actual_int], actual_int, base.LABEL_TO_NAME[actual_int]]
                + predicted
                + [int(value == actual_int) for value in predicted]
            )
            positions[actual_int] += 1


def write_report(payload: dict[str, Any]) -> None:
    results = payload["results"]
    selected_depth = payload["selection"]["configured_max_depth"]
    selected = next(item for item in results if item["configured_max_depth"] == selected_depth)
    current = next(item for item in results if item["configured_max_depth"] == 5)
    unlimited = next(item for item in results if item["configured_max_depth"] is None)
    diagnostic_best = max(
        results,
        key=lambda item: (
            item["purged_final"]["test_macro_f1"],
            item["purged_final"]["test_accuracy"],
            -(item["configured_max_depth"] or 10_000),
        ),
    )
    deeper_results = [
        item for item in results if item["configured_max_depth"] is None or item["configured_max_depth"] > 5
    ]
    deeper_train_gain = max(
        item["purged_final"]["train_accuracy"] for item in deeper_results
    ) - current["purged_final"]["train_accuracy"]
    deeper_best_test_change = max(
        item["purged_final"]["test_accuracy"] for item in deeper_results
    ) - current["purged_final"]["test_accuracy"]

    if deeper_train_gain > 1e-12 and deeper_best_test_change < -1e-12:
        overfit_sentence = (
            "깊이를 5보다 늘렸을 때 학습 정확도는 상승했지만 테스트 정확도는 하락해 "
            "깊은 트리의 과적합 신호가 나타났습니다."
        )
    elif unlimited["purged_final"]["accuracy_gap"] > current["purged_final"]["accuracy_gap"] + 0.02:
        overfit_sentence = (
            "무제한 트리의 학습–테스트 격차가 depth 5보다 커져 깊이를 늘릴수록 "
            "과적합 위험이 증가하는 패턴입니다."
        )
    else:
        overfit_sentence = (
            "depth 5 이후 학습–테스트 격차가 뚜렷하게 확대되지 않아, 이 곡선만으로는 "
            "심한 깊이 과적합 신호가 확인되지 않았습니다."
        )

    metadata = payload["metadata"]
    lines = [
        "# DT 깊이별 과적합 진단",
        "",
        "## 실험 조건",
        "",
        "- 데이터·특성: 기존 `learn.py`와 동일한 200 ms 창 및 15개 특성",
        f"- stride: {STRIDE_MS} ms, 인접 창 overlap {learn.WINDOW_MS - STRIDE_MS} ms",
        "- 최종 분할: 클래스별 시간순 앞 75% 학습, 뒤 25% 테스트",
        f"- purge 분할: 각 클래스의 학습 경계에서 {PURGE_WINDOWS}개 창을 제거해 첫 테스트 창과 원시 샘플을 공유하지 않도록 처리",
        "- 깊이 후보: `1, 2, 3, 4, 5, 6, 7, 8, 10, 12, None`",
        "- 추천 깊이 선택: 최종 테스트셋이 아니라 학습 구간 내부의 purged 검증 macro F1 기준; 동률이면 더 얕은 깊이 선택",
        "",
        "### 데이터 수",
        "",
        f"- 기존 분할: 학습 {metadata['standard_train_windows']}개 / 테스트 {metadata['test_windows']}개",
        f"- 최종 purged 분할: 학습 {metadata['purged_train_windows']}개 / 테스트 {metadata['test_windows']}개 / 제거 {metadata['primary_purged_windows']}개",
        f"- 내부 선택 분할: 적합 {metadata['internal_fit_windows']}개 / 검증 {metadata['internal_validation_windows']}개 / 제거 {metadata['internal_purged_windows']}개",
        "",
        "## 깊이별 결과",
        "",
        "| max_depth | 실제 깊이 | 리프 | 내부 적합 정확도 | 내부 검증 정확도 | 검증 Macro F1 | purged 학습 정확도 | purged 테스트 정확도 | 격차 | 기존 분할 테스트 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        validation = result["internal_validation"]
        purged = result["purged_final"]
        standard = result["standard_final"]
        lines.append(
            f"| {depth_name(result['configured_max_depth'])} | {purged['actual_depth']} | {purged['leaves']} | "
            f"{percent(validation['fit_accuracy'])} | {percent(validation['validation_accuracy'])} | "
            f"{percent(validation['validation_macro_f1'])} | {percent(purged['train_accuracy'])} | "
            f"{percent(purged['test_accuracy'])} | {purged['accuracy_gap'] * 100:+.2f}%p | "
            f"{percent(standard['test_accuracy'])} |"
        )

    lines.extend(
        [
            "",
            "## 해석",
            "",
            f"- 내부 검증으로 선택된 깊이는 **{depth_name(selected_depth)}**입니다. 최종 purged 테스트 정확도는 {percent(selected['purged_final']['test_accuracy'])}, macro F1은 {percent(selected['purged_final']['test_macro_f1'])}입니다.",
            f"- 현재 설정인 depth 5는 학습 정확도 {percent(current['purged_final']['train_accuracy'])}, 테스트 정확도 {percent(current['purged_final']['test_accuracy'])}, 격차 {current['purged_final']['accuracy_gap'] * 100:+.2f}%p입니다.",
            f"- 무제한 트리는 실제 깊이 {unlimited['purged_final']['actual_depth']}, 리프 {unlimited['purged_final']['leaves']}개이며 학습 정확도 {percent(unlimited['purged_final']['train_accuracy'])}, 테스트 정확도 {percent(unlimited['purged_final']['test_accuracy'])}, 격차 {unlimited['purged_final']['accuracy_gap'] * 100:+.2f}%p입니다.",
            f"- {overfit_sentence}",
            f"- 참고로 최종 테스트 곡선에서 가장 높은 조합은 depth {depth_name(diagnostic_best['configured_max_depth'])}의 {percent(diagnostic_best['purged_final']['test_accuracy'])}입니다. 이 값은 테스트셋을 본 사후 진단값이므로 깊이 선택 근거로 사용하지 않았습니다.",
            "",
            "## 한계",
            "",
            "- purge는 분할 경계의 직접적인 원시 샘플 중복만 제거합니다. 학습과 테스트가 여전히 같은 수집 세션에서 왔으므로 세션 특성에 대한 과적합은 측정하지 못합니다.",
            "- 깊이별 테스트 결과를 여러 번 비교하면 테스트셋 자체에 맞춘 선택 편향이 생길 수 있습니다. 그래서 추천 깊이는 내부 검증 결과로만 선택했습니다.",
            "- 최종 판단에는 별도 수집 세션을 통째로 테스트하는 평가가 가장 중요합니다.",
            "",
            "## 생성 파일",
            "",
            "- `dt_depth_summary.csv`: 깊이별 핵심 지표와 일반화 격차",
            "- `dt_depth_results.json`: 분할 정보, 클래스별 지표, 혼동행렬",
            "- `dt_depth_predictions.csv`: purged 최종 테스트의 깊이별 예측",
        ]
    )
    (SCRIPT_DIR / "DT_DEPTH_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    X, y = build_dataset()
    standard_X_train, standard_X_test, standard_y_train, standard_y_test, standard_counts = (
        split_sequential(X, y, purge=False)
    )
    purged_X_train, purged_X_test, purged_y_train, purged_y_test, purged_counts = (
        split_sequential(X, y, purge=True)
    )
    if not np.array_equal(standard_y_test, purged_y_test) or not np.array_equal(
        standard_X_test, purged_X_test
    ):
        raise AssertionError("purge must not change the final test partition")

    X_fit, X_validation, y_fit, y_validation, internal_counts = split_sequential(
        purged_X_train,
        purged_y_train,
        purge=True,
    )
    results: list[dict[str, Any]] = []
    purged_predictions: dict[str, np.ndarray] = {}
    for depth in DEPTHS:
        validation = validation_evaluate(depth, X_fit, y_fit, X_validation, y_validation)
        standard_result, _standard_prediction = fit_and_evaluate(
            depth,
            standard_X_train,
            standard_y_train,
            standard_X_test,
            standard_y_test,
        )
        purged_result, purged_prediction = fit_and_evaluate(
            depth,
            purged_X_train,
            purged_y_train,
            purged_X_test,
            purged_y_test,
        )
        results.append(
            {
                "configured_max_depth": depth,
                "internal_validation": validation,
                "standard_final": standard_result,
                "purged_final": purged_result,
            }
        )
        purged_predictions[depth_name(depth)] = purged_prediction
        print(
            f"depth={depth_name(depth):>4}: train={percent(purged_result['train_accuracy'])}, "
            f"validation={percent(validation['validation_accuracy'])}, "
            f"test={percent(purged_result['test_accuracy'])}, "
            f"actual_depth={purged_result['actual_depth']}, leaves={purged_result['leaves']}"
        )

    selected = max(
        results,
        key=lambda item: (
            item["internal_validation"]["validation_macro_f1"],
            item["internal_validation"]["validation_accuracy"],
            -(item["configured_max_depth"] or 10_000),
        ),
    )
    metadata = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "random_seed": RANDOM_SEED,
        "sample_rate_hz": learn.SAMPLE_RATE_HZ,
        "window_ms": learn.WINDOW_MS,
        "window_samples": learn.WINDOW_SIZE,
        "stride_ms": STRIDE_MS,
        "stride_samples": STEP_SIZE,
        "overlap_ms": learn.WINDOW_MS - STRIDE_MS,
        "train_ratio": TRAIN_RATIO,
        "depth_candidates": list(DEPTHS),
        "feature_count": int(X.shape[1]),
        "feature_names": learn.feature_names(2),
        "total_windows": int(len(X)),
        "standard_train_windows": int(len(standard_X_train)),
        "purged_train_windows": int(len(purged_X_train)),
        "test_windows": int(len(purged_X_test)),
        "primary_purged_windows": int(len(standard_X_train) - len(purged_X_train)),
        "internal_fit_windows": int(len(X_fit)),
        "internal_validation_windows": int(len(X_validation)),
        "internal_purged_windows": int(
            sum(count["purged"] for count in internal_counts.values())
        ),
        "standard_split_counts": standard_counts,
        "purged_split_counts": purged_counts,
        "internal_split_counts": internal_counts,
    }
    payload = {
        "metadata": metadata,
        "selection": {
            "method": "purged_internal_validation_macro_f1_then_accuracy_then_shallower",
            "configured_max_depth": selected["configured_max_depth"],
            "validation_accuracy": selected["internal_validation"]["validation_accuracy"],
            "validation_macro_f1": selected["internal_validation"]["validation_macro_f1"],
            "final_purged_test_accuracy": selected["purged_final"]["test_accuracy"],
            "final_purged_test_macro_f1": selected["purged_final"]["test_macro_f1"],
        },
        "results": results,
    }
    with (SCRIPT_DIR / "dt_depth_results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_summary_csv(results)
    write_predictions(purged_y_test, purged_predictions)
    write_report(payload)
    print(
        f"selected depth={depth_name(selected['configured_max_depth'])}, "
        f"validation_macro_f1={percent(selected['internal_validation']['validation_macro_f1'])}, "
        f"final_test_accuracy={percent(selected['purged_final']['test_accuracy'])}"
    )
    print(f"Wrote DT depth results under: {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
