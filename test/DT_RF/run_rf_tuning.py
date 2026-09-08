"""Tune Random Forest settings without using the final test partitions.

The primary experiment follows ``run_dt_rf_stride.py``: a 200 ms window,
50/100/150/200 ms strides, the project's 15 engineered features, and a
chronological 75/25 train/test split per class.  RF candidates are selected on
an additional chronological 75/25 fit/validation split made only inside the
primary training partition.  Selected settings are then refit on the full
primary training data and evaluated once on the untouched final test data.

All generated artifacts are written next to this script.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterSampler
from sklearn.tree import DecisionTreeClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import learn  # noqa: E402
import run_dt_rf_stride as base  # noqa: E402


RANDOM_SEED = 42
CANDIDATE_COUNT = 32
BASELINE_CONFIG = {
    "n_estimators": 300,
    "criterion": "gini",
    "max_depth": 5,
    "max_features": "sqrt",
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "bootstrap": True,
    "class_weight": None,
}
PARAMETER_SPACE = {
    "n_estimators": [100, 300, 500],
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 3, 5, 8, 12],
    "max_features": ["sqrt", "log2", 0.5, 1.0],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
    "class_weight": [None, "balanced"],
}
VARIANT_NAMES = {
    "dt_reference": "DT reference",
    "rf_baseline": "RF baseline",
    "rf_stride_tuned": "RF stride-tuned",
    "rf_global_tuned": "RF global-tuned",
}


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Convert sampler values to stable Python scalars and key order."""
    return {
        "n_estimators": int(config["n_estimators"]),
        "criterion": str(config["criterion"]),
        "max_depth": None if config["max_depth"] is None else int(config["max_depth"]),
        "max_features": (
            str(config["max_features"])
            if isinstance(config["max_features"], str)
            else float(config["max_features"])
        ),
        "min_samples_split": int(config["min_samples_split"]),
        "min_samples_leaf": int(config["min_samples_leaf"]),
        "bootstrap": bool(config["bootstrap"]),
        "class_weight": (
            None if config["class_weight"] is None else str(config["class_weight"])
        ),
    }


def config_key(config: dict[str, Any]) -> str:
    return json.dumps(config, ensure_ascii=True, sort_keys=True)


def candidate_configs() -> list[dict[str, Any]]:
    """Return baseline plus 31 reproducibly sampled, unique RF settings."""
    candidates = [normalize_config(BASELINE_CONFIG)]
    seen = {config_key(candidates[0])}
    sampled = ParameterSampler(PARAMETER_SPACE, n_iter=96, random_state=RANDOM_SEED)
    for raw_config in sampled:
        config = normalize_config(raw_config)
        key = config_key(config)
        if key not in seen:
            candidates.append(config)
            seen.add(key)
        if len(candidates) == CANDIDATE_COUNT:
            break
    if len(candidates) != CANDIDATE_COUNT:
        raise RuntimeError(f"expected {CANDIDATE_COUNT} unique candidates, got {len(candidates)}")
    return candidates


def make_rf(config: dict[str, Any]) -> RandomForestClassifier:
    return RandomForestClassifier(
        **config,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )


def build_stride_data(stride_ms: int) -> dict[str, Any]:
    step_size = learn.SAMPLE_RATE_HZ * stride_ms // 1000
    datasets: list[tuple[np.ndarray, np.ndarray]] = []
    for _class_name, relative_path, label in learn.CLASSES:
        raw_X, class_y = base.make_raw_dataset(PROJECT_ROOT / relative_path, label, step_size)
        datasets.append((raw_X, class_y))
    raw_X = np.vstack([class_X for class_X, _class_y in datasets])
    y = np.concatenate([class_y for _class_X, class_y in datasets])
    X = learn.engineered_features(raw_X)
    X_train, X_test, y_train, y_test, primary_counts = base.split_sequential_by_class(X, y)
    X_fit, X_val, y_fit, y_val, internal_counts = base.split_sequential_by_class(X_train, y_train)
    return {
        "stride_ms": stride_ms,
        "total_windows": int(len(X)),
        "train_windows": int(len(X_train)),
        "test_windows": int(len(X_test)),
        "fit_windows": int(len(X_fit)),
        "validation_windows": int(len(X_val)),
        "primary_split_counts": primary_counts,
        "internal_split_counts": internal_counts,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_fit": X_fit,
        "X_val": X_val,
        "y_fit": y_fit,
        "y_val": y_val,
    }


def validation_score(config: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    model = make_rf(config)
    start = time.perf_counter()
    model.fit(data["X_fit"], data["y_fit"])
    fit_seconds = time.perf_counter() - start
    prediction = model.predict(data["X_val"])
    return {
        "validation_accuracy": float(accuracy_score(data["y_val"], prediction)),
        "validation_macro_f1": float(
            f1_score(
                data["y_val"],
                prediction,
                labels=base.CLASS_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "fit_seconds": float(fit_seconds),
        "mean_tree_depth": float(np.mean([tree.get_depth() for tree in model.estimators_])),
        "mean_tree_leaves": float(np.mean([tree.get_n_leaves() for tree in model.estimators_])),
    }


def paired_comparison(
    first_prediction: np.ndarray,
    second_prediction: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, int]:
    first_correct = first_prediction == y_test
    second_correct = second_prediction == y_test
    return {
        "both_correct": int(np.sum(first_correct & second_correct)),
        "first_only_correct": int(np.sum(first_correct & ~second_correct)),
        "second_only_correct": int(np.sum(~first_correct & second_correct)),
        "both_wrong": int(np.sum(~first_correct & ~second_correct)),
    }


def evaluate_variant(
    model: Any,
    config: dict[str, Any] | None,
    data: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    result, prediction, fitted = base.evaluate_model(
        model,
        data["X_train"],
        data["y_train"],
        data["X_test"],
        data["y_test"],
    )
    if config is None:
        result["configuration"] = "DecisionTreeClassifier(max_depth=5, random_state=42)"
        result["model_complexity"] = {
            "depth": int(fitted.get_depth()),
            "leaves": int(fitted.get_n_leaves()),
            "nodes": int(fitted.tree_.node_count),
        }
    else:
        result["configuration"] = config
        result["model_complexity"] = {
            "trees": int(len(fitted.estimators_)),
            "mean_depth": float(np.mean([tree.get_depth() for tree in fitted.estimators_])),
            "mean_leaves": float(np.mean([tree.get_n_leaves() for tree in fitted.estimators_])),
        }
    return result, prediction


def format_setting(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    return str(value)


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_validation_csv(records: list[dict[str, Any]]) -> None:
    fields = [
        "stride_ms",
        "candidate_id",
        *BASELINE_CONFIG.keys(),
        "validation_accuracy_percent",
        "validation_macro_f1_percent",
        "fit_seconds",
        "mean_tree_depth",
        "mean_tree_leaves",
        "selected_for_stride",
        "selected_global",
    ]
    with (SCRIPT_DIR / "rf_validation_candidates.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {
                "stride_ms": record["stride_ms"],
                "candidate_id": record["candidate_id"],
                **record["config"],
                "validation_accuracy_percent": f'{record["validation_accuracy"] * 100:.4f}',
                "validation_macro_f1_percent": f'{record["validation_macro_f1"] * 100:.4f}',
                "fit_seconds": f'{record["fit_seconds"]:.6f}',
                "mean_tree_depth": f'{record["mean_tree_depth"]:.4f}',
                "mean_tree_leaves": f'{record["mean_tree_leaves"]:.4f}',
                "selected_for_stride": int(record["selected_for_stride"]),
                "selected_global": int(record["selected_global"]),
            }
            writer.writerow(row)


def write_final_csv(final_results: list[dict[str, Any]]) -> None:
    fields = [
        "stride_ms",
        "variant",
        "accuracy_percent",
        "balanced_accuracy_percent",
        "macro_precision_percent",
        "macro_recall_percent",
        "macro_f1_percent",
        "fit_seconds",
        "prediction_microseconds_per_window",
        *BASELINE_CONFIG.keys(),
    ]
    with (SCRIPT_DIR / "rf_final_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for stride_result in final_results:
            for variant, result in stride_result["variants"].items():
                configuration = result["configuration"] if isinstance(result["configuration"], dict) else {}
                writer.writerow(
                    {
                        "stride_ms": stride_result["stride_ms"],
                        "variant": VARIANT_NAMES[variant],
                        "accuracy_percent": f'{result["accuracy"] * 100:.4f}',
                        "balanced_accuracy_percent": f'{result["balanced_accuracy"] * 100:.4f}',
                        "macro_precision_percent": f'{result["macro_precision"] * 100:.4f}',
                        "macro_recall_percent": f'{result["macro_recall"] * 100:.4f}',
                        "macro_f1_percent": f'{result["macro_f1"] * 100:.4f}',
                        "fit_seconds": f'{result["fit_seconds"]:.6f}',
                        "prediction_microseconds_per_window": f'{result["prediction_microseconds_per_window"]:.6f}',
                        **configuration,
                    }
                )


def write_predictions(
    stride_ms: int,
    y_test: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    path = SCRIPT_DIR / f"rf_tuning_predictions_{stride_ms}ms.csv"
    variants = list(VARIANT_NAMES)
    positions = {label: 0 for label in base.CLASS_LABELS}
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["test_row", "class_test_position", "actual_label", "actual_class"]
            + [f"{variant}_prediction" for variant in variants]
            + [f"{variant}_correct" for variant in variants]
        )
        for row_index, actual in enumerate(y_test):
            actual_int = int(actual)
            predicted = [int(predictions[variant][row_index]) for variant in variants]
            writer.writerow(
                [row_index, positions[actual_int], actual_int, base.LABEL_TO_NAME[actual_int]]
                + predicted
                + [int(value == actual_int) for value in predicted]
            )
            positions[actual_int] += 1


def config_markdown(config: dict[str, Any]) -> str:
    return ", ".join(f"{key}={format_setting(value)}" for key, value in config.items())


def write_report(payload: dict[str, Any]) -> None:
    final_results = payload["final_results"]
    global_selection = payload["global_selection"]
    lines = [
        "# Random Forest 설정 탐색 결과",
        "",
        "## 실험 설계",
        "",
        "- 데이터·특성: 기존 `learn.py`와 동일한 200 ms 창 및 15개 특성",
        "- stride: 50/100/150/200 ms",
        "- 최종 분할: 클래스별 시간순 앞 75% 학습, 뒤 25% 테스트",
        "- 설정 선택: 75% 학습 구간 내부를 다시 시간순 75/25 적합·검증으로 분할",
        f"- 후보: baseline 1개와 고정 시드 무작위 조합 31개, 총 {CANDIDATE_COUNT}개",
        "- 선택 기준: 검증 macro F1, 동률이면 검증 정확도",
        "- 최종 테스트셋은 설정 선택에 사용하지 않음",
        "- DT reference: `DecisionTreeClassifier(max_depth=5, random_state=42)`",
        "",
        "### 탐색 범위",
        "",
        "| 설정 | 후보값 |",
        "|---|---|",
    ]
    for key, values in PARAMETER_SPACE.items():
        lines.append(f"| `{key}` | " + ", ".join(f"`{format_setting(value)}`" for value in values) + " |")

    lines.extend(
        [
            "",
            "## stride별 선택 설정",
            "",
            "| stride | 후보 | 검증 Macro F1 | n_estimators | criterion | max_depth | max_features | min_split | min_leaf | bootstrap | class_weight |",
            "|---:|---:|---:|---:|---|---:|---|---:|---:|---|---|",
        ]
    )
    for result in final_results:
        selection = result["stride_selection"]
        config = selection["config"]
        lines.append(
            f"| {result['stride_ms']} ms | {selection['candidate_id']} | {percent(selection['validation_macro_f1'])} | "
            f"{config['n_estimators']} | {config['criterion']} | {format_setting(config['max_depth'])} | "
            f"{format_setting(config['max_features'])} | {config['min_samples_split']} | "
            f"{config['min_samples_leaf']} | {config['bootstrap']} | {format_setting(config['class_weight'])} |"
        )

    lines.extend(
        [
            "",
            "### 전체 stride 공통 설정",
            "",
            f"네 stride의 검증 macro F1 평균이 가장 높은 후보는 **{global_selection['candidate_id']}번**입니다.",
            "",
            f"`{config_markdown(global_selection['config'])}`",
            "",
            f"평균 검증 macro F1은 {percent(global_selection['mean_validation_macro_f1'])}, 가장 낮은 stride의 검증 macro F1은 {percent(global_selection['min_validation_macro_f1'])}입니다.",
            "",
            "## 최종 테스트 결과",
            "",
            "| stride | 학습/테스트 창 | DT reference | RF baseline | RF stride-tuned | 변화 | RF global-tuned |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in final_results:
        variants = result["variants"]
        baseline = variants["rf_baseline"]
        tuned = variants["rf_stride_tuned"]
        lines.append(
            f"| {result['stride_ms']} ms | {result['train_windows']}/{result['test_windows']} | "
            f"{percent(variants['dt_reference']['accuracy'])} | {percent(baseline['accuracy'])} | "
            f"{percent(tuned['accuracy'])} | {(tuned['accuracy'] - baseline['accuracy']) * 100:+.2f}%p | "
            f"{percent(variants['rf_global_tuned']['accuracy'])} |"
        )

    lines.extend(
        [
            "",
            "### Macro F1",
            "",
            "| stride | DT reference | RF baseline | RF stride-tuned | 변화 | RF global-tuned |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in final_results:
        variants = result["variants"]
        baseline = variants["rf_baseline"]
        tuned = variants["rf_stride_tuned"]
        lines.append(
            f"| {result['stride_ms']} ms | {percent(variants['dt_reference']['macro_f1'])} | "
            f"{percent(baseline['macro_f1'])} | {percent(tuned['macro_f1'])} | "
            f"{(tuned['macro_f1'] - baseline['macro_f1']) * 100:+.2f}%p | "
            f"{percent(variants['rf_global_tuned']['macro_f1'])} |"
        )

    lines.extend(
        [
            "",
            "## 클래스별 F1",
            "",
            "| stride | 모델 | " + " | ".join(base.CLASS_NAMES) + " |",
            "|---:|---|" + "---:|" * len(base.CLASS_NAMES),
        ]
    )
    for result in final_results:
        for variant, variant_result in result["variants"].items():
            values = [
                percent(variant_result["classification_report"][name]["f1-score"])
                for name in base.CLASS_NAMES
            ]
            lines.append(
                f"| {result['stride_ms']} ms | {VARIANT_NAMES[variant]} | "
                + " | ".join(values)
                + " |"
            )

    improvements = [
        (
            result["stride_ms"],
            result["variants"]["rf_stride_tuned"]["accuracy"]
            - result["variants"]["rf_baseline"]["accuracy"],
        )
        for result in final_results
    ]
    best_gain = max(improvements, key=lambda item: item[1])
    lines.extend(
        [
            "",
            "## 해석 및 주의사항",
            "",
            f"- baseline 대비 가장 큰 테스트 정확도 변화는 {best_gain[0]} ms의 {best_gain[1] * 100:+.2f}%p입니다.",
            "- stride별 최적 설정은 내부 검증셋 기준으로 선택했습니다. 최종 테스트 성능이 항상 baseline보다 좋아진다는 보장은 없으며, 나빠진 경우도 그대로 기록합니다.",
            "- 후보 수가 32개이므로 탐색하지 않은 조합이 많습니다. 이 결과는 해당 탐색 공간의 제한된 무작위 탐색 결과입니다.",
            "- 짧은 stride의 겹치는 창은 독립 표본이 아니며, 설정 탐색 과정에서도 같은 수집 세션의 시간적 변화에 과적합될 수 있습니다.",
            "- 실제 모델 선정 전에는 별도 수집 세션이나 사용자 단위의 외부 검증이 필요합니다.",
            "- RF의 트리 수·깊이·특성 수 증가는 임베디드 메모리와 추론 비용을 늘릴 수 있으므로 정확도와 별도로 배포 비용을 확인해야 합니다.",
            "",
            "## 생성 파일",
            "",
            "- `rf_validation_candidates.csv`: 4개 stride × 32개 후보의 내부 검증 결과",
            "- `rf_final_summary.csv`: DT·baseline RF·stride별 튜닝 RF·공통 튜닝 RF의 최종 테스트 결과",
            "- `rf_tuning_results.json`: 설정, 세부 지표, 혼동행렬, 선택 정보",
            "- `rf_tuning_predictions_*ms.csv`: 최종 테스트 창별 네 모델 예측",
        ]
    )
    (SCRIPT_DIR / "RF_TUNING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    candidates = candidate_configs()
    stride_data = {stride: build_stride_data(stride) for stride in base.STRIDES_MS}
    validation_records: list[dict[str, Any]] = []
    stride_selections: dict[int, dict[str, Any]] = {}

    for stride_ms, data in stride_data.items():
        stride_records: list[dict[str, Any]] = []
        for candidate_id, config in enumerate(candidates):
            score = validation_score(config, data)
            record = {
                "stride_ms": stride_ms,
                "candidate_id": candidate_id,
                "config": config,
                **score,
            }
            validation_records.append(record)
            stride_records.append(record)
        selected = max(
            stride_records,
            key=lambda item: (
                item["validation_macro_f1"],
                item["validation_accuracy"],
                -item["candidate_id"],
            ),
        )
        stride_selections[stride_ms] = selected
        print(
            f"{stride_ms:>3} ms search: candidate={selected['candidate_id']}, "
            f"validation_macro_f1={percent(selected['validation_macro_f1'])}"
        )

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in validation_records:
        grouped[record["candidate_id"]].append(record)
    aggregates = []
    for candidate_id, records in sorted(grouped.items()):
        aggregates.append(
            {
                "candidate_id": candidate_id,
                "config": candidates[candidate_id],
                "mean_validation_macro_f1": float(
                    np.mean([record["validation_macro_f1"] for record in records])
                ),
                "min_validation_macro_f1": float(
                    np.min([record["validation_macro_f1"] for record in records])
                ),
                "mean_validation_accuracy": float(
                    np.mean([record["validation_accuracy"] for record in records])
                ),
            }
        )
    global_selection = max(
        aggregates,
        key=lambda item: (
            item["mean_validation_macro_f1"],
            item["min_validation_macro_f1"],
            item["mean_validation_accuracy"],
            -item["candidate_id"],
        ),
    )

    for record in validation_records:
        record["selected_for_stride"] = (
            record["candidate_id"]
            == stride_selections[record["stride_ms"]]["candidate_id"]
        )
        record["selected_global"] = record["candidate_id"] == global_selection["candidate_id"]

    final_results: list[dict[str, Any]] = []
    for stride_ms, data in stride_data.items():
        selected = stride_selections[stride_ms]
        model_specs = {
            "dt_reference": (
                DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED),
                None,
            ),
            "rf_baseline": (make_rf(BASELINE_CONFIG), normalize_config(BASELINE_CONFIG)),
            "rf_stride_tuned": (make_rf(selected["config"]), selected["config"]),
            "rf_global_tuned": (
                make_rf(global_selection["config"]),
                global_selection["config"],
            ),
        }
        variants: dict[str, dict[str, Any]] = {}
        predictions: dict[str, np.ndarray] = {}
        for variant, (model, config) in model_specs.items():
            result, prediction = evaluate_variant(model, config, data)
            variants[variant] = result
            predictions[variant] = prediction
        write_predictions(stride_ms, data["y_test"], predictions)
        final_results.append(
            {
                "stride_ms": stride_ms,
                "total_windows": data["total_windows"],
                "train_windows": data["train_windows"],
                "test_windows": data["test_windows"],
                "fit_windows": data["fit_windows"],
                "validation_windows": data["validation_windows"],
                "primary_split_counts": data["primary_split_counts"],
                "internal_split_counts": data["internal_split_counts"],
                "stride_selection": {
                    "candidate_id": selected["candidate_id"],
                    "config": selected["config"],
                    "validation_accuracy": selected["validation_accuracy"],
                    "validation_macro_f1": selected["validation_macro_f1"],
                },
                "variants": variants,
                "paired_stride_tuned_vs_baseline": paired_comparison(
                    predictions["rf_baseline"],
                    predictions["rf_stride_tuned"],
                    data["y_test"],
                ),
                "paired_stride_tuned_vs_dt": paired_comparison(
                    predictions["dt_reference"],
                    predictions["rf_stride_tuned"],
                    data["y_test"],
                ),
            }
        )
        print(
            f"{stride_ms:>3} ms final: baseline={percent(variants['rf_baseline']['accuracy'])}, "
            f"stride-tuned={percent(variants['rf_stride_tuned']['accuracy'])}, "
            f"global-tuned={percent(variants['rf_global_tuned']['accuracy'])}"
        )

    payload = {
        "metadata": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
            "random_seed": RANDOM_SEED,
            "sample_rate_hz": learn.SAMPLE_RATE_HZ,
            "window_ms": learn.WINDOW_MS,
            "strides_ms": list(base.STRIDES_MS),
            "primary_train_ratio": base.TRAIN_RATIO,
            "internal_fit_ratio": base.TRAIN_RATIO,
            "selection_metric": "validation_macro_f1_then_accuracy",
            "candidate_count": CANDIDATE_COUNT,
            "class_order": base.CLASS_NAMES,
            "feature_names": learn.feature_names(2),
        },
        "parameter_space": PARAMETER_SPACE,
        "candidates": [
            {"candidate_id": candidate_id, "config": config}
            for candidate_id, config in enumerate(candidates)
        ],
        "validation_results": validation_records,
        "candidate_aggregates": aggregates,
        "global_selection": global_selection,
        "final_results": final_results,
    }
    with (SCRIPT_DIR / "rf_tuning_results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_validation_csv(validation_records)
    write_final_csv(final_results)
    write_report(payload)
    print(f"Wrote RF tuning results under: {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
