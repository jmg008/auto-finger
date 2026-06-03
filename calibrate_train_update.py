import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


SAMPLE_RATE_HZ = 1000
WINDOW_MS = 200
STEP_MS = 50
RAW_FEATURE_COUNT = 6
NORMALIZATION_MIN_DENOM = 0.001
BALANCE_EPSILON = 0.001

CLASSES = [
    ("rock", "rock.txt", 0),
    ("paper", "paper.txt", 1),
    ("none", "none.txt", 2),
]

LABEL_NAMES = {
    0: "rock",
    1: "paper",
    2: "none",
}

LABEL_NAMES_INV = {name: label for label, name in LABEL_NAMES.items()}
CLASS_LABELS = sorted(LABEL_NAMES)


class TreeNode:
    def __init__(self, label, feature=None, threshold=None, left=None, right=None):
        self.label = int(label)
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    @property
    def is_leaf(self):
        return self.feature is None


class SimpleDecisionTree:
    def __init__(self, max_depth=5, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def fit(self, X, y):
        self.root = self._build(X, y, depth=0)
        return self

    def predict(self, X):
        return np.array([self.predict_one(row) for row in X])

    def predict_one(self, row):
        node = self.root
        while not node.is_leaf:
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.label

    def _build(self, X, y, depth):
        label = majority_label(y)

        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return TreeNode(label)

        split = best_split(X, y, self.min_samples_leaf)
        if split is None:
            return TreeNode(label)

        feature, threshold, left_mask = split
        right_mask = ~left_mask
        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[right_mask], y[right_mask], depth + 1)
        return TreeNode(label, feature=feature, threshold=threshold, left=left, right=right)

COLLECT_ORDER = [
    ("none", "none.txt", 2),
    ("rock", "rock.txt", 0),
    ("paper", "paper.txt", 1),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect EMG serial data, train a decision tree, and update Arduino code."
    )
    parser.add_argument("--port", default="COM15", help="Serial port, for example COM15")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--seconds", type=float, default=8.0, help="Seconds to collect per class")
    parser.add_argument("--sets", type=int, default=1, help="Number of none/rock/paper collection sets")
    parser.add_argument("--settle", type=float, default=2.0, help="Seconds to wait before recording")
    parser.add_argument("--skip-collect", action="store_true", help="Use existing rock/paper/none txt files")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--split-mode", choices=("sequential", "random"), default="sequential")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE_HZ)
    parser.add_argument("--window-ms", type=int, default=WINDOW_MS)
    parser.add_argument("--step-ms", type=int, default=STEP_MS)
    parser.add_argument("--ino", default="EMG_Classify_Serial/EMG_Classify_Serial.ino")
    parser.add_argument("--header", default="EMG_Model.h")
    parser.add_argument("--arduino-cli", default="arduino-cli", help="Path to arduino-cli executable")
    parser.add_argument("--fqbn", default="arduino:avr:uno", help="Arduino board FQBN, for example arduino:avr:uno")
    parser.add_argument("--sensor-sketch", default="EMG_Sensor", help="Raw EMG serial sketch folder")
    parser.add_argument(
        "--classify-sketch",
        default=None,
        help="Classifier sketch folder. Defaults to the parent folder of --ino.",
    )
    parser.add_argument("--no-upload", action="store_true", help="Disable all arduino-cli compile/upload steps")
    parser.add_argument("--no-sensor-upload", action="store_true", help="Do not upload the raw sensor sketch")
    parser.add_argument("--no-classify-upload", action="store_true", help="Do not upload the classifier sketch")
    parser.add_argument("--upload-wait", type=float, default=2.0, help="Seconds to wait after each upload")
    return parser.parse_args()


def run_command(command, cwd=None):
    print("$ " + " ".join(str(part) for part in command))
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)

    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip())

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def resolve_arduino_cli(cli_name):
    path = Path(cli_name)
    if path.exists():
        return str(path)

    found = shutil.which(cli_name)
    if found:
        return found

    candidates = [
        Path.home() / "Documents" / "Codex" / "tools" / "arduino-cli" / "arduino-cli.exe",
        Path.home() / "AppData" / "Local" / "Programs" / "Arduino CLI" / "arduino-cli.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "arduino-cli executable was not found. Pass the full path with "
        "--arduino-cli, for example: --arduino-cli "
        r"C:\Users\j\Documents\Codex\tools\arduino-cli\arduino-cli.exe"
    )


def normalize_sketch_path(sketch_path):
    path = Path(sketch_path)
    if path.suffix.lower() == ".ino":
        path = path.parent
    return path


def upload_sketch(args, sketch_path, label):
    path = normalize_sketch_path(sketch_path)
    if not path.exists():
        raise FileNotFoundError(f"{label} sketch not found: {path}")

    print(f"\n[{label}] Compile")
    run_command([args.arduino_cli, "compile", "--fqbn", args.fqbn, str(path)])

    print(f"\n[{label}] Upload")
    run_command([args.arduino_cli, "upload", "-p", args.port, "--fqbn", args.fqbn, str(path)])

    if args.upload_wait > 0:
        print(f"Waiting {args.upload_wait:.1f}s after upload...")
        time.sleep(args.upload_wait)


def ensure_pyserial_available():
    try:
        import serial  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("pyserial is required for collection: pip install pyserial") from exc


def parse_sensor_line(raw_line):
    line = raw_line.decode("utf-8", errors="ignore").strip()
    if not line:
        return None

    parts = line.split(",")
    if len(parts) < 2:
        return None

    try:
        return int(float(parts[0])), int(float(parts[1]))
    except ValueError:
        return None


def collect_one_class(ser, name, out_path, seconds, settle, append=False, set_index=1, set_count=1):
    input(f"\nPrepare '{name}' ({set_index}/{set_count}), then press Enter to collect.")
    ser.reset_input_buffer()

    if settle > 0:
        print(f"Settling for {settle:.1f}s...")
        time.sleep(settle)
        ser.reset_input_buffer()

    rows = []
    end_at = time.monotonic() + seconds
    next_report = time.monotonic() + 1.0

    while time.monotonic() < end_at:
        parsed = parse_sensor_line(ser.readline())
        if parsed is not None:
            rows.append(parsed)

        if time.monotonic() >= next_report:
            print(f"  {name}: {len(rows)} samples")
            next_report += 1.0

    if not rows:
        raise RuntimeError(
            "No numeric sensor rows were collected. Upload the raw serial sketch first; "
            "it should print lines like '123,456'."
        )

    mode = "a" if append else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        for inside, outside in rows:
            f.write(f"{inside},{outside}\n")

    action = "Appended" if append else "Saved"
    print(f"{action} {len(rows)} samples to {out_path}")


def collect_data(args):
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("pyserial is required for collection: pip install pyserial") from exc

    print("Upload the raw EMG serial sketch before collecting.")
    print("Expected serial format: inside,outside")

    with serial.Serial(port=args.port, baudrate=args.baud, timeout=0.1) as ser:
        time.sleep(2.0)
        ser.reset_input_buffer()

        written_files = set()
        for set_index in range(1, args.sets + 1):
            print(f"\n[Collection set {set_index}/{args.sets}]")
            for name, filename, _label in COLLECT_ORDER:
                path = Path(filename)
                collect_one_class(
                    ser,
                    name,
                    path,
                    args.seconds,
                    args.settle,
                    append=path in written_files,
                    set_index=set_index,
                    set_count=args.sets,
                )
                written_files.add(path)


def load_sensor_data(path, window_size):
    data = np.genfromtxt(path, delimiter=",")

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if np.isnan(data).any():
        data = data[~np.isnan(data).any(axis=1)]

    if len(data) < window_size:
        raise ValueError(f"{path}: need at least {window_size} samples, got {len(data)}")

    return data.astype(float)


def raw_window_features(window):
    mav = np.mean(np.abs(window), axis=0)
    rms = np.sqrt(np.mean(np.square(window), axis=0))
    wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
    return np.concatenate([mav, rms, wl])


def compute_normalization_params(raw_X, y):
    rest_X = raw_X[y == LABEL_NAMES_INV["none"]]
    if len(rest_X) == 0:
        raise ValueError("none class data is required to compute rest normalization values")

    rest_values = np.percentile(rest_X, 50, axis=0)
    calib_values = np.percentile(raw_X, 95, axis=0)
    calib_values = np.maximum(calib_values, rest_values + NORMALIZATION_MIN_DENOM)
    return rest_values, calib_values


def normalized_features(raw_X, rest_values, calib_values):
    denom = np.maximum(calib_values - rest_values, NORMALIZATION_MIN_DENOM)
    normalized = (raw_X - rest_values) / denom
    normalized = np.maximum(normalized, 0.0)

    if normalized.shape[1] != RAW_FEATURE_COUNT:
        return normalized

    mav_diff = normalized[:, 0] - normalized[:, 1]
    rms_diff = normalized[:, 2] - normalized[:, 3]
    wl_diff = normalized[:, 4] - normalized[:, 5]

    mav_sum = normalized[:, 0] + normalized[:, 1]
    rms_sum = normalized[:, 2] + normalized[:, 3]
    wl_sum = normalized[:, 4] + normalized[:, 5]

    mav_balance = mav_diff / (mav_sum + BALANCE_EPSILON)
    rms_balance = rms_diff / (rms_sum + BALANCE_EPSILON)
    wl_balance = wl_diff / (wl_sum + BALANCE_EPSILON)

    relation_features = np.column_stack([
        mav_diff,
        rms_diff,
        wl_diff,
        mav_sum,
        rms_sum,
        wl_sum,
        mav_balance,
        rms_balance,
        wl_balance,
    ])

    return np.column_stack([normalized, relation_features])


def make_raw_dataset(path, label, window_size, step_size):
    data = load_sensor_data(path, window_size)
    features = []
    labels = []

    for start in range(0, len(data) - window_size + 1, step_size):
        window = data[start:start + window_size]
        features.append(raw_window_features(window))
        labels.append(label)

    return np.array(features), np.array(labels)


def feature_names(channel_count):
    if channel_count == 2:
        return [
            "MAV_ch1_norm",
            "MAV_ch2_norm",
            "RMS_ch1_norm",
            "RMS_ch2_norm",
            "WL_ch1_norm",
            "WL_ch2_norm",
            "MAV_diff",
            "RMS_diff",
            "WL_diff",
            "MAV_sum",
            "RMS_sum",
            "WL_sum",
            "MAV_balance",
            "RMS_balance",
            "WL_balance",
        ]

    names = []
    for metric in ("MAV", "RMS", "WL"):
        for channel in range(channel_count):
            names.append(f"{metric}_ch{channel + 1}_norm")
    return names


def majority_label(y):
    counts = np.bincount(y.astype(int), minlength=max(CLASS_LABELS) + 1)
    return int(np.argmax(counts))


def gini_from_counts(counts):
    total = np.sum(counts)
    if total == 0:
        return 0.0

    probabilities = counts / total
    return 1.0 - np.sum(probabilities * probabilities)


def best_split(X, y, min_samples_leaf):
    sample_count, feature_count = X.shape
    parent_counts = np.bincount(y.astype(int), minlength=max(CLASS_LABELS) + 1)
    parent_impurity = gini_from_counts(parent_counts)
    best_gain = 0.0
    best_feature = None
    best_threshold = None

    for feature in range(feature_count):
        order = np.argsort(X[:, feature], kind="mergesort")
        values = X[order, feature]
        labels = y[order].astype(int)

        left_counts = np.zeros(max(CLASS_LABELS) + 1, dtype=float)
        right_counts = parent_counts.astype(float).copy()

        for index in range(1, sample_count):
            moved_label = labels[index - 1]
            left_counts[moved_label] += 1
            right_counts[moved_label] -= 1

            if values[index] == values[index - 1]:
                continue

            left_size = index
            right_size = sample_count - index
            if left_size < min_samples_leaf or right_size < min_samples_leaf:
                continue

            left_impurity = gini_from_counts(left_counts)
            right_impurity = gini_from_counts(right_counts)
            weighted_impurity = (
                left_size * left_impurity + right_size * right_impurity
            ) / sample_count
            gain = parent_impurity - weighted_impurity

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = (values[index - 1] + values[index]) / 2.0

    if best_feature is None:
        return None

    left_mask = X[:, best_feature] <= best_threshold
    return best_feature, best_threshold, left_mask


def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred):
    matrix = np.zeros((len(CLASS_LABELS), len(CLASS_LABELS)), dtype=int)
    for actual, predicted in zip(y_true.astype(int), y_pred.astype(int)):
        matrix[actual, predicted] += 1
    return matrix


def print_classification_report(y_true, y_pred):
    print("label     precision  recall  f1-score  support")
    for label in CLASS_LABELS:
        name = LABEL_NAMES[label]
        true_positive = np.sum((y_true == label) & (y_pred == label))
        predicted_positive = np.sum(y_pred == label)
        actual_positive = np.sum(y_true == label)

        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"{name:<9} {precision:9.2f} {recall:7.2f} {f1:9.2f} {actual_positive:8d}")


def train_test_split_by_class(X, y, train_ratio=0.8, split_mode="sequential", random_state=42):
    train_indices = []
    test_indices = []
    rng = np.random.default_rng(random_state)

    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        if split_mode == "random":
            indices = indices.copy()
            rng.shuffle(indices)

        split_at = int(len(indices) * train_ratio)

        if split_at == 0 or split_at == len(indices):
            raise ValueError(f"label {label}: cannot split {len(indices)} windows")

        train_indices.extend(indices[:split_at])
        test_indices.extend(indices[split_at:])

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def train_model(args):
    window_size = args.sample_rate * args.window_ms // 1000
    step_size = args.sample_rate * args.step_ms // 1000

    datasets = []
    for name, path, label in CLASSES:
        class_raw_X, class_y = make_raw_dataset(path, label, window_size, step_size)
        datasets.append((class_raw_X, class_y))
        print(f"{name}: {len(class_raw_X)} windows")

    raw_X = np.vstack([class_raw_X for class_raw_X, _ in datasets])
    y = np.concatenate([class_y for _, class_y in datasets])
    rest_values, calib_values = compute_normalization_params(raw_X, y)
    X = normalized_features(raw_X, rest_values, calib_values)

    X_train, X_test, y_train, y_test = train_test_split_by_class(
        X,
        y,
        split_mode=args.split_mode,
        random_state=args.random_state,
    )

    model_factory, model_name = get_model_factory(args.max_depth)
    print(f"Model trainer: {model_name}")
    print(f"Validation split: {args.split_mode}")

    eval_clf = model_factory()
    eval_clf.fit(X_train, y_train)
    y_pred = eval_clf.predict(X_test)

    print("\n[Validation]")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print_classification_report(y_test, y_pred)

    final_clf = model_factory()
    final_clf.fit(X, y)

    channel_count = load_sensor_data(CLASSES[0][1], window_size).shape[1]
    names = feature_names(channel_count)
    return final_clf, names, window_size, step_size, rest_values, calib_values


def get_model_factory(max_depth):
    try:
        from sklearn.tree import DecisionTreeClassifier

        def factory():
            return DecisionTreeClassifier(max_depth=max_depth, random_state=42)

        return factory, "scikit-learn"
    except ImportError:
        def factory():
            return SimpleDecisionTree(max_depth=max_depth)

        return factory, "internal simple CART"


def float_literal(value):
    text = f"{float(value):.10g}"
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return f"{text}f"


def float_array_literal(values):
    return "{ " + ", ".join(float_literal(value) for value in values) + " }"


def export_simple_node(node, indent="    "):
    if node.is_leaf:
        return f"{indent}return {node.label};\n"

    code = f"{indent}if (x[{node.feature}] <= {float_literal(node.threshold)}) {{\n"
    code += export_simple_node(node.left, indent + "    ")
    code += f"{indent}}} else {{\n"
    code += export_simple_node(node.right, indent + "    ")
    code += f"{indent}}}\n"
    return code


def export_sklearn_node(clf, node_id, indent="    "):
    tree = clf.tree_
    feature = tree.feature[node_id]

    if feature == -2:
        label = int(np.argmax(tree.value[node_id][0]))
        return f"{indent}return {label};\n"

    threshold = tree.threshold[node_id]
    left = tree.children_left[node_id]
    right = tree.children_right[node_id]

    code = f"{indent}if (x[{feature}] <= {float_literal(threshold)}) {{\n"
    code += export_sklearn_node(clf, left, indent + "    ")
    code += f"{indent}}} else {{\n"
    code += export_sklearn_node(clf, right, indent + "    ")
    code += f"{indent}}}\n"
    return code


def export_node(clf):
    if hasattr(clf, "tree_"):
        return export_sklearn_node(clf, 0)

    return export_simple_node(clf.root)


def export_predict_function(clf):
    code = "int predictEMG(float *x) {\n"
    code += export_node(clf)
    code += "}\n"
    return code

def export_header(predict_code, feature_names_text, window_size, step_size):
    lines = [
        "#pragma once",
        "",
        "// Generated by calibrate_train_update.py",
        f"// Window samples: {window_size}",
        f"// Step samples: {step_size}",
        "// Raw normalization: rest = none median, calib = overall 95th percentile",
        f"// Feature order: {feature_names_text}",
        "",
        predict_code.rstrip(),
        "",
    ]
    return "\n".join(lines)


def find_function_bounds(source, signature_pattern):
    match = re.search(signature_pattern, source)
    if not match:
        raise RuntimeError("Could not find predictEMG(float *x) in the Arduino sketch")

    brace_start = source.find("{", match.start())
    if brace_start == -1:
        raise RuntimeError("Could not find function opening brace")

    depth = 0
    for index in range(brace_start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return match.start(), index + 1

    raise RuntimeError("Could not find function closing brace")


def replace_array_constant(source, name, values):
    pattern = rf"const float {name}\[RAW_FEATURE_COUNT\] = \{{[^}}]*\}};"
    replacement = f"const float {name}[RAW_FEATURE_COUNT] = {float_array_literal(values)};"
    updated, count = re.subn(pattern, replacement, source)
    if count != 1:
        raise RuntimeError(f"Could not update {name} in the Arduino sketch")
    return updated


def replace_predict_function(ino_path, predict_code, window_size, step_size, rest_values, calib_values):
    path = Path(ino_path)
    source = path.read_text(encoding="utf-8")

    start, end = find_function_bounds(source, r"int\s+predictEMG\s*\(\s*float\s*\*\s*x\s*\)")
    source = source[:start] + predict_code.rstrip() + source[end:]

    source = re.sub(r"const int WINDOW_SIZE = \d+;", f"const int WINDOW_SIZE = {window_size};", source)
    source = re.sub(r"const int STEP_SIZE = \d+;", f"const int STEP_SIZE = {step_size};", source)
    source = re.sub(r"const int FEATURE_COUNT = \d+;", "const int FEATURE_COUNT = 15;", source)
    source = replace_array_constant(source, "NORMALIZATION_REST", rest_values)
    source = replace_array_constant(source, "NORMALIZATION_CALIB", calib_values)

    path.write_text(source, encoding="utf-8")
    print(f"Updated {path}")


def main():
    args = parse_args()
    if args.sets < 1:
        raise ValueError("--sets must be at least 1")

    if not args.no_upload:
        args.arduino_cli = resolve_arduino_cli(args.arduino_cli)

    classify_sketch = args.classify_sketch or str(Path(args.ino).parent)

    if not args.skip_collect:
        ensure_pyserial_available()

        if not args.no_upload and not args.no_sensor_upload:
            upload_sketch(args, args.sensor_sketch, "Raw sensor sketch")

        collect_data(args)

    clf, names, window_size, step_size, rest_values, calib_values = train_model(args)
    predict_code = export_predict_function(clf)
    feature_names_text = ", ".join(names)

    header_text = export_header(predict_code, feature_names_text, window_size, step_size)
    header_path = Path(args.header)
    header_path.write_text(header_text, encoding="utf-8")
    print(f"Updated {header_path}")

    ino_path = Path(args.ino)
    replace_predict_function(ino_path, predict_code, window_size, step_size, rest_values, calib_values)

    sketch_header_path = ino_path.parent / header_path.name
    if sketch_header_path.resolve() != header_path.resolve():
        shutil.copyfile(header_path, sketch_header_path)
        print(f"Copied {header_path} to {sketch_header_path}")

    if not args.no_upload and not args.no_classify_upload:
        upload_sketch(args, classify_sketch, "Classifier sketch")

    print("\nDone.")
    print("Feature order:", feature_names_text)
    print("Labels: 0=rock, 1=paper, 2=none")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)
