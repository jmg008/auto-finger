import numpy as np
from micromlgen import port
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier


SAMPLE_RATE_HZ = 1000
WINDOW_MS = 200
STEP_MS = 50

WINDOW_SIZE = SAMPLE_RATE_HZ * WINDOW_MS // 1000
STEP_SIZE = SAMPLE_RATE_HZ * STEP_MS // 1000
RAW_FEATURE_COUNT = 6
NORMALIZATION_MIN_DENOM = 0.001
BALANCE_EPSILON = 0.001

ROCK_LABEL = 0   # jwieot-eul ttae
PAPER_LABEL = 1  # pyeot-eul ttae
NONE_LABEL = 2   # him an jun jungnip sangtae

CLASSES = [
    ("rock", "rock.txt", ROCK_LABEL),
    ("paper", "paper.txt", PAPER_LABEL),
    ("none", "none.txt", NONE_LABEL),
]


def load_sensor_data(path):
    data = np.genfromtxt(path, delimiter=",")

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if np.isnan(data).any():
        data = data[~np.isnan(data).any(axis=1)]

    if len(data) < WINDOW_SIZE:
        raise ValueError(f"{path}: {WINDOW_SIZE} samples are required, but only {len(data)} found")

    return data.astype(float)


def raw_window_features(window):
    mav = np.mean(np.abs(window), axis=0)
    rms = np.sqrt(np.mean(np.square(window), axis=0))
    wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
    return np.concatenate([mav, rms, wl])


def normalized_features(raw_X, rest_values, calib_values):
    denom = np.maximum(calib_values - rest_values, NORMALIZATION_MIN_DENOM)
    normalized = np.maximum((raw_X - rest_values) / denom, 0.0)

    mav_diff = normalized[:, 0] - normalized[:, 1]
    rms_diff = normalized[:, 2] - normalized[:, 3]
    wl_diff = normalized[:, 4] - normalized[:, 5]

    mav_sum = normalized[:, 0] + normalized[:, 1]
    rms_sum = normalized[:, 2] + normalized[:, 3]
    wl_sum = normalized[:, 4] + normalized[:, 5]

    relation_features = np.column_stack([
        mav_diff,
        rms_diff,
        wl_diff,
        mav_sum,
        rms_sum,
        wl_sum,
        mav_diff / (mav_sum + BALANCE_EPSILON),
        rms_diff / (rms_sum + BALANCE_EPSILON),
        wl_diff / (wl_sum + BALANCE_EPSILON),
    ])

    return np.column_stack([normalized, relation_features])


def make_raw_dataset(path, label):
    data = load_sensor_data(path)
    features = []
    labels = []

    for start in range(0, len(data) - WINDOW_SIZE + 1, STEP_SIZE):
        window = data[start:start + WINDOW_SIZE]
        features.append(raw_window_features(window))
        labels.append(label)

    return np.array(features), np.array(labels)


def train_test_split_by_class(X, y, train_ratio=0.8):
    train_indices = []
    test_indices = []

    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        split_at = int(len(indices) * train_ratio)

        if split_at == 0 or split_at == len(indices):
            raise ValueError(f"label {label}: train/test split is not possible with {len(indices)} windows")

        train_indices.extend(indices[:split_at])
        test_indices.extend(indices[split_at:])

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


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


def main():
    datasets = []
    for name, path, label in CLASSES:
        class_raw_X, class_y = make_raw_dataset(path, label)
        datasets.append((class_raw_X, class_y))
        print(f"{name} windows: {len(class_raw_X)}")

    raw_X = np.vstack([class_raw_X for class_raw_X, _ in datasets])
    y = np.concatenate([class_y for _, class_y in datasets])
    rest_X = raw_X[y == NONE_LABEL]
    rest_values = np.percentile(rest_X, 50, axis=0)
    calib_values = np.percentile(raw_X, 95, axis=0)
    calib_values = np.maximum(calib_values, rest_values + NORMALIZATION_MIN_DENOM)
    X = normalized_features(raw_X, rest_values, calib_values)

    X_train, X_test, y_train, y_test = train_test_split_by_class(X, y)

    channel_count = load_sensor_data("rock.txt").shape[1]
    names = feature_names(channel_count)

    print(f"Sample rate: {SAMPLE_RATE_HZ} Hz")
    print(f"Window: {WINDOW_MS} ms ({WINDOW_SIZE} samples)")
    print(f"Step: {STEP_MS} ms ({STEP_SIZE} samples)")
    print(f"Features: {', '.join(names)}")
    print("Normalization rest:", ", ".join(f"{value:.10g}" for value in rest_values))
    print("Normalization calib:", ", ".join(f"{value:.10g}" for value in calib_values))
    print(f"Train windows: {len(X_train)}, Test windows: {len(X_test)}")

    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\n[Result]")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[name for name, _, _ in CLASSES]))

    c_code = port(clf, classname="EMG_DecisionTree")

    with open("EMG_Model.h", "w", encoding="utf-8") as f:
        f.write(c_code)

    print("\nDone: EMG_Model.h was generated.")
    print("Input feature order:", ", ".join(names))


if __name__ == "__main__":
    main()
