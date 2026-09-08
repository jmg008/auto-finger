"""Compare six classifiers using the project's existing EMG training method.

The data loading, 200 ms windowing, 100 ms stride, and 15 engineered features
come directly from ``learn.py``.  Each class is split chronologically, with
the first 75% of windows used for training and the final 25% for testing.

GA-SVM tunes RBF-SVM C and gamma on a chronological validation subset made
only from the training partition, then refits the selected model on all of the
training data.  No test observation is used for model or hyperparameter choice.

Every generated artifact is written next to this script.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
import time
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import learn  # noqa: E402  (project module, imported after PROJECT_ROOT is set)


RANDOM_SEED = 42
TRAIN_RATIO = 0.75
GA_VALIDATION_RATIO = 0.25
CLASS_NAMES = [name for name, _path, _label in learn.CLASSES]
CLASS_LABELS = [label for _name, _path, label in learn.CLASSES]
LABEL_TO_NAME = dict(zip(CLASS_LABELS, CLASS_NAMES))
DISPLAY_NAMES = {
    "dt": "DT",
    "rf": "RF",
    "lda": "LDA",
    "ga_svm": "GA-SVM",
    "mlp_1x9": "MLP-1×9",
    "mlp_2x9": "MLP-2×9",
}


@dataclass(frozen=True)
class GASettings:
    population_size: int = 24
    generations: int = 18
    elite_count: int = 4
    tournament_size: int = 3
    mutation_probability: float = 0.30
    mutation_sigma: float = 0.45
    log10_c_min: float = -3.0
    log10_c_max: float = 3.0
    log10_gamma_min: float = -5.0
    log10_gamma_max: float = 1.0
    random_seed: int = RANDOM_SEED


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_sequential_by_class(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, int]]]:
    """Split each class in time order without shuffling."""
    train_indices: list[int] = []
    test_indices: list[int] = []
    split_counts: dict[str, dict[str, int]] = {}

    for label in CLASS_LABELS:
        indices = np.flatnonzero(y == label)
        split_at = int(len(indices) * train_ratio)
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

    train_array = np.asarray(train_indices, dtype=int)
    test_array = np.asarray(test_indices, dtype=int)
    return X[train_array], X[test_array], y[train_array], y[test_array], split_counts


def make_svm(log10_c: float, log10_gamma: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    C=10.0**log10_c,
                    gamma=10.0**log10_gamma,
                ),
            ),
        ]
    )


def tune_ga_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    settings: GASettings,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Tune log10(C) and log10(gamma) with a small reproducible GA."""
    X_fit, X_val, y_fit, y_val, validation_counts = split_sequential_by_class(
        X_train, y_train, 1.0 - GA_VALIDATION_RATIO
    )
    rng = np.random.default_rng(settings.random_seed)
    low = np.asarray([settings.log10_c_min, settings.log10_gamma_min], dtype=float)
    high = np.asarray([settings.log10_c_max, settings.log10_gamma_max], dtype=float)
    population = rng.uniform(low, high, size=(settings.population_size, 2))

    # Seed several conventional points without changing the population size.
    seeds = np.asarray([[0.0, -2.0], [1.0, -2.0], [2.0, -3.0], [-1.0, -1.0]])
    population[: len(seeds)] = np.clip(seeds, low, high)

    cache: dict[tuple[float, float], tuple[float, float]] = {}
    history: list[dict[str, Any]] = []
    global_best: tuple[tuple[float, float], np.ndarray] | None = None

    def evaluate(individual: np.ndarray) -> tuple[float, float]:
        key = (round(float(individual[0]), 10), round(float(individual[1]), 10))
        if key not in cache:
            model = make_svm(*individual)
            model.fit(X_fit, y_fit)
            prediction = model.predict(X_val)
            cache[key] = (
                float(f1_score(y_val, prediction, labels=CLASS_LABELS, average="macro", zero_division=0)),
                float(accuracy_score(y_val, prediction)),
            )
        return cache[key]

    def tournament(scored: list[tuple[tuple[float, float], np.ndarray]]) -> np.ndarray:
        chosen = rng.integers(0, len(scored), size=settings.tournament_size)
        return max((scored[int(index)] for index in chosen), key=lambda item: item[0])[1]

    for generation in range(settings.generations):
        scored = [(evaluate(individual), individual.copy()) for individual in population]
        scored.sort(key=lambda item: item[0], reverse=True)
        if global_best is None or scored[0][0] > global_best[0]:
            global_best = (scored[0][0], scored[0][1].copy())

        history.append(
            {
                "generation": generation + 1,
                "best_validation_macro_f1": float(scored[0][0][0]),
                "best_validation_accuracy": float(scored[0][0][1]),
                "mean_validation_macro_f1": float(np.mean([item[0][0] for item in scored])),
                "best_log10_c": float(scored[0][1][0]),
                "best_log10_gamma": float(scored[0][1][1]),
            }
        )

        if generation == settings.generations - 1:
            break

        next_population = [item[1].copy() for item in scored[: settings.elite_count]]
        while len(next_population) < settings.population_size:
            parent_a = tournament(scored)
            parent_b = tournament(scored)
            alpha = rng.uniform(-0.25, 1.25, size=2)
            child = alpha * parent_a + (1.0 - alpha) * parent_b
            mutation_mask = rng.random(2) < settings.mutation_probability
            child += mutation_mask * rng.normal(0.0, settings.mutation_sigma, size=2)
            next_population.append(np.clip(child, low, high))
        population = np.asarray(next_population)

    assert global_best is not None
    best_score, best_individual = global_best
    details = {
        "selected_log10_c": float(best_individual[0]),
        "selected_log10_gamma": float(best_individual[1]),
        "selected_c": float(10.0 ** best_individual[0]),
        "selected_gamma": float(10.0 ** best_individual[1]),
        "validation_macro_f1": float(best_score[0]),
        "validation_accuracy": float(best_score[1]),
        "unique_candidates_evaluated": len(cache),
        "fit_windows": int(len(X_fit)),
        "validation_windows": int(len(X_val)),
        "validation_split_counts": validation_counts,
        "settings": asdict(settings),
    }
    return details, history


def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, Any]:
    captured_warnings: list[str] = []
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=ConvergenceWarning)
        fitted_model = model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    captured_warnings.extend(str(item.message) for item in caught)

    start = time.perf_counter()
    prediction = fitted_model.predict(X_test)
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
        "prediction_microseconds_per_window": float(predict_seconds / len(X_test) * 1_000_000.0),
        "warnings": captured_warnings,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=CLASS_LABELS)
        .astype(int)
        .tolist(),
    }
    return result, prediction.astype(int), fitted_model


def model_details(model_name: str, model: Any) -> dict[str, Any]:
    if model_name == "dt":
        return {
            "configuration": "DecisionTreeClassifier(max_depth=5, random_state=42)",
            "depth": int(model.get_depth()),
            "leaves": int(model.get_n_leaves()),
            "nodes": int(model.tree_.node_count),
        }
    if model_name == "rf":
        return {
            "configuration": "RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42, n_jobs=1)",
            "trees": int(len(model.estimators_)),
            "mean_depth": float(np.mean([tree.get_depth() for tree in model.estimators_])),
            "mean_leaves": float(np.mean([tree.get_n_leaves() for tree in model.estimators_])),
        }
    if model_name == "lda":
        return {
            "configuration": 'LinearDiscriminantAnalysis(solver="svd")',
            "classes": int(len(model.classes_)),
            "discriminant_components": int(model.scalings_.shape[1]),
        }
    if model_name.startswith("mlp"):
        mlp = model.named_steps["mlp"]
        return {
            "configuration": repr(mlp),
            "hidden_layer_sizes": list(mlp.hidden_layer_sizes),
            "iterations": int(mlp.n_iter_),
            "parameter_count": int(
                sum(weights.size for weights in mlp.coefs_)
                + sum(bias.size for bias in mlp.intercepts_)
            ),
        }
    if model_name == "ga_svm":
        svc = model.named_steps["svc"]
        return {
            "configuration": repr(svc),
            "support_vectors": int(np.sum(svc.n_support_)),
            "support_vectors_by_class": {
                name: int(count) for name, count in zip(CLASS_NAMES, svc.n_support_)
            },
        }
    raise KeyError(model_name)


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_summary_csv(results: dict[str, dict[str, Any]]) -> None:
    path = SCRIPT_DIR / "summary.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
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
        ranked = sorted(
            results.items(),
            key=lambda item: (item[1]["accuracy"], item[1]["macro_f1"]),
            reverse=True,
        )
        for rank, (model_name, result) in enumerate(ranked, start=1):
            writer.writerow(
                [
                    rank,
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


def write_confusion_csv(model_name: str, matrix: list[list[int]]) -> None:
    with (SCRIPT_DIR / f"confusion_{model_name}.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([class_name, *row])


def write_predictions_csv(y_test: np.ndarray, predictions: dict[str, np.ndarray]) -> None:
    positions = {label: 0 for label in CLASS_LABELS}
    with (SCRIPT_DIR / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        model_names = list(predictions)
        writer.writerow(
            ["test_row", "class_test_position", "actual_label", "actual_class"]
            + [f"{name}_prediction" for name in model_names]
            + [f"{name}_correct" for name in model_names]
        )
        for row_index, actual in enumerate(y_test):
            actual_int = int(actual)
            predicted = [int(predictions[name][row_index]) for name in model_names]
            writer.writerow(
                [row_index, positions[actual_int], actual_int, LABEL_TO_NAME[actual_int]]
                + predicted
                + [int(value == actual_int) for value in predicted]
            )
            positions[actual_int] += 1


def write_report(payload: dict[str, Any]) -> None:
    results = payload["results"]
    ranked = sorted(
        results.items(),
        key=lambda item: (item[1]["accuracy"], item[1]["macro_f1"]),
        reverse=True,
    )
    best_name, best = ranked[0]
    ga = payload["ga_svm_tuning"]
    lines = [
        "# EMG 분류 모델 6종 비교 결과",
        "",
        "## 실험 조건",
        "",
        "- 데이터: 프로젝트 루트의 `none.txt`, `rock.txt`, `paper.txt`, `middle.txt`, `thumb.txt`",
        f"- 기존 `learn.py` 방식: {learn.SAMPLE_RATE_HZ:,} Hz, {learn.WINDOW_MS} ms 창({learn.WINDOW_SIZE}개 샘플), {learn.STEP_MS} ms stride",
        "- 특성: MAV/RMS/WL 6개와 2채널 차이·합·balance 9개, 총 15개",
        "- 분할: 클래스별 시간 순서를 유지해 앞 75%를 학습, 뒤 25%를 테스트",
        "- 공정성: 여섯 모델 모두 완전히 동일한 학습/테스트 창과 특성을 사용",
        "- 표준화: 스케일에 민감한 GA-SVM과 MLP에만 `StandardScaler`를 적용하고, scaler는 학습 데이터로만 적합",
        "- 재현성: 난수 시드 42 고정",
        "",
        "### 모델 설정",
        "",
        "- DT: `DecisionTreeClassifier(max_depth=5, random_state=42)` — 기존 설정과 동일",
        "- RF: `RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)`",
        "- LDA: `LinearDiscriminantAnalysis(solver=\"svd\")`",
        f"- GA-SVM: RBF SVM의 C와 gamma를 유전 알고리즘으로 탐색; 선택값 C={ga['selected_c']:.8g}, gamma={ga['selected_gamma']:.8g}",
        "- MLP-1×9: 은닉층 `(9,)`, ReLU, L-BFGS, 최대 5,000회",
        "- MLP-2×9: 은닉층 `(9, 9)`, ReLU, L-BFGS, 최대 5,000회",
        "",
        "GA-SVM은 최종 테스트셋을 보지 않습니다. 최종 테스트용 25%를 제외한 학습 구간 안에서 클래스별 앞 75%를 GA 적합용, 뒤 25%를 검증용으로 사용해 macro F1을 최대화한 뒤, 선택된 C/gamma로 전체 75% 학습 구간을 다시 학습했습니다.",
        "",
        "## 데이터 분할",
        "",
        "| class | 원본 행 | 전체 창 | 학습 창 | 테스트 창 |",
        "|---|---:|---:|---:|---:|",
    ]
    for class_name in CLASS_NAMES:
        source = payload["metadata"]["sources"][class_name]
        counts = payload["metadata"]["split_counts"][class_name]
        lines.append(
            f"| {class_name} | {source['valid_rows']} | {counts['total']} | {counts['train']} | {counts['test']} |"
        )
    metadata = payload["metadata"]
    lines.extend(
        [
            f"| **합계** | **{metadata['total_source_rows']}** | **{metadata['total_windows']}** | **{metadata['train_windows']}** | **{metadata['test_windows']}** |",
            "",
            "## 성능 요약",
            "",
            "| 순위 | 모델 | 정확도 | Balanced accuracy | Macro precision | Macro recall | Macro F1 |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, (model_name, result) in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {DISPLAY_NAMES[model_name]} | {percent(result['accuracy'])} | "
            f"{percent(result['balanced_accuracy'])} | {percent(result['macro_precision'])} | "
            f"{percent(result['macro_recall'])} | {percent(result['macro_f1'])} |"
        )

    lines.extend(
        [
            "",
            f"이 고정 테스트셋에서는 **{DISPLAY_NAMES[best_name]}**가 정확도 {percent(best['accuracy'])}, macro F1 {percent(best['macro_f1'])}로 가장 높았습니다.",
            "",
            "## 클래스별 F1",
            "",
            "| 모델 | " + " | ".join(CLASS_NAMES) + " |",
            "|---|" + "---:|" * len(CLASS_NAMES),
        ]
    )
    for model_name, result in ranked:
        class_f1 = [percent(result["classification_report"][name]["f1-score"]) for name in CLASS_NAMES]
        lines.append(
            f"| {DISPLAY_NAMES[model_name]} | " + " | ".join(class_f1) + " |"
        )

    lines.extend(
        [
            "",
            "## 혼동행렬",
            "",
            "각 표의 행은 실제 클래스, 열은 예측 클래스입니다.",
        ]
    )
    for model_name, result in ranked:
        lines.extend(
            [
                "",
                f"### {DISPLAY_NAMES[model_name]}",
                "",
                "| actual \\ predicted | " + " | ".join(CLASS_NAMES) + " |",
                "|---|" + "---:|" * len(CLASS_NAMES),
            ]
        )
        for class_name, row in zip(CLASS_NAMES, result["confusion_matrix"]):
            lines.append(f"| {class_name} | " + " | ".join(str(value) for value in row) + " |")

    hardest_class = min(
        CLASS_NAMES,
        key=lambda name: np.mean(
            [results[model_name]["classification_report"][name]["f1-score"] for model_name in results]
        ),
    )
    lines.extend(
        [
            "",
            "## 해석 및 주의사항",
            "",
            f"- 여섯 모델의 평균 클래스별 F1을 기준으로 가장 어려운 클래스는 `{hardest_class}`였습니다.",
            "- 현재 프로젝트의 C 헤더 생성 경로와 바로 호환되는 것은 DT입니다. 다른 모델의 배포 난이도·메모리·추론시간은 이 정확도 표와 별도로 판단해야 합니다.",
            "- 학습/추론 시간은 이 PC에서 한 번 측정한 참고값이며, 임베디드 장치의 실행시간을 뜻하지 않습니다. 원시 값은 `summary.csv`와 `results.json`에 기록했습니다.",
            "- 인접 200 ms 창은 100 ms를 공유합니다. 시간순 분할 경계에서도 마지막 학습 창과 첫 테스트 창이 100 ms를 공유하므로 정확도가 다소 낙관적일 수 있습니다. 이는 기존 `learn.py`의 창 생성 후 분할 방식을 그대로 따른 결과입니다.",
            "- 한 번의 고정 holdout 결과이므로 통계적 우월성을 확정하지 않습니다. 실제 선택 전에는 다른 수집 세션·사용자 단위의 완전 분리 평가가 필요합니다.",
            "",
            "## 생성 파일",
            "",
            "- `results.json`: 전체 설정, GA 탐색 이력, 세부 지표",
            "- `summary.csv`: 모델별 핵심 지표와 측정 시간",
            "- `predictions.csv`: 테스트 창별 실제값, 예측값, 정오표",
            "- `confusion_*.csv`: 모델별 혼동행렬",
        ]
    )
    (SCRIPT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    datasets: list[tuple[np.ndarray, np.ndarray]] = []
    sources: dict[str, dict[str, Any]] = {}
    channel_counts: set[int] = set()

    for class_name, relative_path, label in learn.CLASSES:
        source_path = PROJECT_ROOT / relative_path
        sensor_data = learn.load_sensor_data(source_path)
        raw_X, class_y = learn.make_raw_dataset(source_path, label)
        datasets.append((raw_X, class_y))
        channel_counts.add(int(sensor_data.shape[1]))
        sources[class_name] = {
            "path": relative_path,
            "valid_rows": int(len(sensor_data)),
            "channels": int(sensor_data.shape[1]),
            "sha256": sha256_file(source_path),
        }

    if channel_counts != {2}:
        raise ValueError(f"expected the existing two-channel method, got {channel_counts}")

    raw_X = np.vstack([class_X for class_X, _class_y in datasets])
    y = np.concatenate([class_y for _class_X, class_y in datasets])
    X = learn.engineered_features(raw_X)
    X_train, X_test, y_train, y_test, split_counts = split_sequential_by_class(
        X, y, TRAIN_RATIO
    )

    assert X.shape[1] == len(learn.feature_names(2)) == 15
    for class_name, counts in split_counts.items():
        assert counts["train"] == math.floor(counts["total"] * TRAIN_RATIO), class_name
        assert counts["first_test_window"] == counts["train"], class_name

    ga_start = time.perf_counter()
    ga_details, ga_history = tune_ga_svm(X_train, y_train, GASettings())
    ga_tuning_seconds = time.perf_counter() - ga_start
    ga_details["tuning_seconds"] = float(ga_tuning_seconds)

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
            ("lda", LinearDiscriminantAnalysis(solver="svd")),
            (
                "ga_svm",
                make_svm(ga_details["selected_log10_c"], ga_details["selected_log10_gamma"]),
            ),
            (
                "mlp_1x9",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "mlp",
                            MLPClassifier(
                                hidden_layer_sizes=(9,),
                                activation="relu",
                                solver="lbfgs",
                                alpha=0.0001,
                                max_iter=5000,
                                random_state=RANDOM_SEED,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "mlp_2x9",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "mlp",
                            MLPClassifier(
                                hidden_layer_sizes=(9, 9),
                                activation="relu",
                                solver="lbfgs",
                                alpha=0.0001,
                                max_iter=5000,
                                random_state=RANDOM_SEED,
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )

    results: dict[str, dict[str, Any]] = OrderedDict()
    predictions: dict[str, np.ndarray] = OrderedDict()
    for model_name, model in models.items():
        result, prediction, fitted_model = evaluate_model(
            model, X_train, y_train, X_test, y_test
        )
        result["model_details"] = model_details(model_name, fitted_model)
        results[model_name] = result
        predictions[model_name] = prediction
        print(
            f"{model_name:>7}: accuracy={percent(result['accuracy'])}, "
            f"macro_f1={percent(result['macro_f1'])}"
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
            "stride_ms": learn.STEP_MS,
            "stride_samples": learn.STEP_SIZE,
            "window_overlap_ms": learn.WINDOW_MS - learn.STEP_MS,
            "train_ratio": TRAIN_RATIO,
            "split_mode": "sequential_by_class_after_windowing",
            "class_order": CLASS_NAMES,
            "class_labels": CLASS_LABELS,
            "sources": sources,
            "total_source_rows": int(sum(source["valid_rows"] for source in sources.values())),
            "split_counts": split_counts,
            "feature_count": int(X.shape[1]),
            "feature_names": learn.feature_names(2),
            "total_windows": int(len(X)),
            "train_windows": int(len(X_train)),
            "test_windows": int(len(X_test)),
        },
        "ga_svm_tuning": ga_details,
        "ga_svm_history": ga_history,
        "results": results,
    }

    with (SCRIPT_DIR / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_summary_csv(results)
    write_predictions_csv(y_test, predictions)
    for model_name, result in results.items():
        write_confusion_csv(model_name, result["confusion_matrix"])
    write_report(payload)

    print(
        f"GA-SVM: C={ga_details['selected_c']:.8g}, gamma={ga_details['selected_gamma']:.8g}, "
        f"validation_macro_f1={percent(ga_details['validation_macro_f1'])}"
    )
    print(f"train/test windows: {len(X_train)}/{len(X_test)}")
    print(f"Wrote results under: {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
