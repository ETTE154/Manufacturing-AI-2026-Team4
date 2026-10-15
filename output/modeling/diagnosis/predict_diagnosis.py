"""Score current 5-second observations with saved electrical-candidate models.

Usage (project environment):
  python predict_diagnosis.py --input observations.csv --output scores.csv

One output row per input observation and model. All original input columns and a
zero-based input_row_index are preserved. Scores concern weak electrical-quality
candidates / reconstruction anomalies; they are not verified fault probabilities.
"""
from pathlib import Path
import argparse
import gzip
import json
import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier

from diagnosis_experiment import AE, SENSORS, TRANSFER

HERE = Path(__file__).resolve().parent
MODEL_NAMES = (
    "xgb_rule_distillation",
    "xgb_rule_sensor_excluded_transfer",
    "ae_unsupervised_all_training",
    "ae_weak_normal_proxy",
)
RESULT_COLUMNS = (
    "input_row_index", "model", "score", "alarm", "threshold",
    "label_type", "score_type", "threshold_basis",
)


def load_saved_models(names):
    """Load local trained artifacts and validate feature/threshold metadata."""
    metadata = json.loads((HERE / "diagnosis_model_metrics.json").read_text(encoding="utf-8"))
    loaded = {}
    for name in names:
        if name not in metadata:
            raise ValueError(f"Saved metadata is missing model {name}")
        info = metadata[name]
        features = info["features"]
        expected = list(TRANSFER if name == "xgb_rule_sensor_excluded_transfer" else SENSORS)
        if features != expected:
            raise ValueError(f"{name}: saved feature order differs from the training definition")
        metrics = info.get("test", info.get("test_weak_proxy_agreement"))
        threshold = float(metrics["threshold"])
        if not np.isfinite(threshold):
            raise ValueError(f"{name}: invalid saved threshold")
        label_type = info["label_info"]["type"]
        if not isinstance(label_type, str) or not label_type:
            raise ValueError(f"{name}: invalid saved label type")
        if name.startswith("xgb"):
            model = XGBClassifier()
            model.load_model(HERE / f"{name}.json")
            booster = model.get_booster()
            if booster.feature_names != features or booster.num_features() != len(features):
                raise ValueError(f"{name}: native model feature metadata mismatch")
            best_iteration = booster.attr("best_iteration")
            if best_iteration is None or int(best_iteration) != int(info["best_iteration"]):
                raise ValueError(f"{name}: native model best iteration mismatch")
            if not 0 <= threshold <= 1:
                raise ValueError(f"{name}: candidate probability threshold outside [0,1]")
            loaded[name] = {
                "model": model, "features": features, "threshold": threshold,
                "label_type": label_type, "score_type": "weak_candidate_probability",
                "threshold_basis": info["threshold_selection"],
            }
        else:
            checkpoint = torch.load(HERE / f"{name}.pt", map_location="cpu", weights_only=True)
            expected_architecture = [len(features), 32, 12, 4, 12, 32, len(features)]
            if checkpoint["features"] != features:
                raise ValueError(f"{name}: checkpoint feature order mismatch")
            if checkpoint["architecture"] != expected_architecture or info["architecture"] != expected_architecture:
                raise ValueError(f"{name}: checkpoint architecture mismatch")
            if float(checkpoint["threshold"]) != threshold:
                raise ValueError(f"{name}: checkpoint threshold mismatch")
            with (HERE / f"{name}_scaler.pkl").open("rb") as handle:
                scaler = pickle.load(handle)
            if list(scaler.feature_names_in_) != features or scaler.n_features_in_ != len(features):
                raise ValueError(f"{name}: scaler feature metadata mismatch")
            model = AE(len(features)).cpu()
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval()
            loaded[name] = {
                "model": model, "scaler": scaler, "features": features,
                "threshold": threshold, "label_type": label_type,
                "score_type": "mean_squared_standardized_reconstruction_error",
                "threshold_basis": info["threshold_selection"],
            }
    return loaded


def validate_chunk(frame, required_features, start_row):
    """Check measurements without dropping, imputing, sorting or altering keys."""
    if len(frame) == 0:
        raise ValueError("Input contains no observation rows")
    collisions = sorted(set(RESULT_COLUMNS).intersection(frame.columns))
    if collisions:
        raise ValueError("Input has reserved output columns: " + ", ".join(collisions))
    required = ["module(equipment)", "localtime", *required_features]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError("Input is missing required columns: " + ", ".join(missing))
    for key in ["module(equipment)", "localtime"]:
        bad = frame[key].isna() | frame[key].astype(str).str.strip().eq("")
        if bad.any():
            index = start_row + int(np.flatnonzero(bad.to_numpy())[0])
            raise ValueError(f"Missing identity {key} at input row {index}")
    measured = frame[required_features].apply(pd.to_numeric, errors="coerce")
    values = measured.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        row, col = np.argwhere(~np.isfinite(values))[0]
        raise ValueError(
            f"Missing/non-numeric/non-finite sensor {required_features[int(col)]} "
            f"at input row {start_row + int(row)}"
        )
    return measured


def score_chunk(frame, models, start_row=0):
    """Return a long table with original row identity and frozen model alarms."""
    required_features = list(dict.fromkeys(
        feature for saved in models.values() for feature in saved["features"]
    ))
    measured = validate_chunk(frame, required_features, start_row)
    parts = []
    for name, saved in models.items():
        values = measured[saved["features"]]
        if name.startswith("xgb"):
            # Keep exact native float32 values in float64 before threshold comparison.
            # A float32 comparison rounds a nextafter threshold and can flip ties.
            scores = np.asarray(saved["model"].predict_proba(values)[:, 1], dtype=np.float64)
        else:
            transformed = saved["scaler"].transform(values).astype(np.float32)
            if not np.isfinite(transformed).all():
                raise ValueError(f"{name}: scaler produced non-finite values")
            tensor = torch.from_numpy(transformed)
            with torch.no_grad():
                errors = (saved["model"](tensor) - tensor).pow(2).numpy()
            scores = errors.mean(axis=1).astype(np.float64)
        if scores.shape != (len(frame),) or not np.isfinite(scores).all():
            raise ValueError(f"{name}: invalid inference scores")
        part = frame.copy()
        part["input_row_index"] = np.arange(start_row, start_row + len(frame), dtype=np.int64)
        part["model"] = name
        part["score"] = scores
        part["alarm"] = (scores >= saved["threshold"]).astype(np.int8)
        part["threshold"] = saved["threshold"]
        part["label_type"] = saved["label_type"]
        part["score_type"] = saved["score_type"]
        part["threshold_basis"] = saved["threshold_basis"]
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values("input_row_index", kind="stable")


def predict_csv(input_path, output_path, names=MODEL_NAMES, chunksize=20000):
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if input_path == output_path:
        raise ValueError("Output must differ from the input observation file")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not (str(output_path).lower().endswith(".csv") or str(output_path).lower().endswith(".csv.gz")):
        raise ValueError("Output must end in .csv or .csv.gz")
    if output_path.exists():
        raise ValueError(f"Output already exists; choose a new path: {output_path}")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    models = load_saved_models(names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".diagnosis-", suffix=".tmp", dir=output_path.parent)
    os.close(fd)
    temporary = Path(temp_name)
    total = 0
    try:
        opener = gzip.open if str(output_path).lower().endswith(".gz") else open
        with opener(temporary, "wt", encoding="utf-8", newline="") as handle:
            reader = pd.read_csv(
                input_path, chunksize=chunksize,
                dtype={"module(equipment)": "string", "localtime": "string",
                       "timestamp": "string", "dt": "string"},
            )
            for frame in reader:
                result = score_chunk(frame, models, total)
                result.to_csv(handle, index=False, header=(total == 0), float_format="%.17g")
                total += len(frame)
        if total == 0:
            raise ValueError("Input contains no observation rows")
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "input": str(input_path), "output": str(output_path),
        "input_observations": total, "output_rows": total * len(models),
        "models": list(models),
        "label_type": {name: saved["label_type"] for name, saved in models.items()},
        "interpretation": "Current electrical-quality candidate / reconstruction scores, not verified fault probabilities",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="New 5-second observation CSV or CSV.gz")
    parser.add_argument("--output", required=True, help="New score table path (.csv or .csv.gz)")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--chunksize", type=int, default=20000)
    args = parser.parse_args()
    if len(args.models) != len(set(args.models)):
        parser.error("Each model may be selected only once")
    try:
        summary = predict_csv(args.input, args.output, args.models, args.chunksize)
    except (ValueError, FileNotFoundError, pd.errors.EmptyDataError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

