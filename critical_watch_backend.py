from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss
)
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

# ----------------------------- Configuration -----------------------------

CSV_PATH = "icu_hourly.csv"
VITALS = [
    "heart_rate", "temperature", "respiratory_rate",
    "systolic_bp", "spo2", "wbc"
]
WINDOW_HOURS = 24
PREDICTION_HOURS = 6
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


# --------------------------- Patient-level split --------------------------

def split_patients(df):
    """Split unique patients, not individual rows, to prevent leakage."""
    patients = df["patient_id"].drop_duplicates().to_numpy()
    train_ids, temp_ids = train_test_split(
        patients, test_size=0.30, random_state=SEED
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.50, random_state=SEED
    )
    train = df[df.patient_id.isin(train_ids)].copy()
    val = df[df.patient_id.isin(val_ids)].copy()
    test = df[df.patient_id.isin(test_ids)].copy()
    return train, val, test


# ------------------------------ Preprocessing -----------------------------

class VitalPreprocessor:
    """
    Fit medians and z-score statistics on TRAINING patients only.
    Adds missingness masks and hours-since-last-measurement channels.
    """

    def fit(self, train_df):
        raw = train_df[VITALS].copy()
        self.medians = raw.median().fillna(0.0)
        filled = raw.fillna(self.medians)
        self.means = filled.mean()
        self.stds = filled.std().replace(0, 1).fillna(1)
        return self

    def transform(self, df):
        pieces = []
        for patient_id, group in df.groupby("patient_id", sort=False):
            group = group.sort_values("hour").copy()
            raw = group[VITALS].copy()

            # 1 means the value was missing before imputation.
            masks = raw.isna().astype(float)
            # Carry a reading forward for at most six hourly rows.
            filled = raw.ffill(limit=6).fillna(self.medians)

            # Time since the last actual measurement, capped at 6 hours.
            deltas = {}
            for col in VITALS:
                last_seen = group["hour"].where(raw[col].notna()).ffill()
                deltas[col + "_delta"] = (
                    group["hour"] - last_seen
                ).fillna(6).clip(0, 6).to_numpy(dtype=float)

            # Physiological clipping, followed by training-set z-score.
            bounds = {
                "heart_rate": (20, 250), "temperature": (30, 43),
                "respiratory_rate": (4, 60), "systolic_bp": (40, 300),
                "spo2": (50, 100), "wbc": (0, 100)
            }
            for col, (lo, hi) in bounds.items():
                filled[col] = filled[col].clip(lo, hi)

            scaled = (filled - self.means) / self.stds
            scaled.columns = [c + "_z" for c in VITALS]

            out = pd.concat(
                [
                    group[["patient_id", "hour", "sepsis_onset_hour"]].reset_index(drop=True),
                    scaled.reset_index(drop=True),
                    masks.add_suffix("_mask").reset_index(drop=True),
                    pd.DataFrame(deltas).reset_index(drop=True),
                ],
                axis=1,
            )
            pieces.append(out)
        return pd.concat(pieces, ignore_index=True)


# -------------------------- Labels and features ---------------------------

SEQ_CHANNELS = (
    [v + "_z" for v in VITALS]
    + [v + "_mask" for v in VITALS]
    + [v + "_delta" for v in VITALS]
)


def make_label(hour, onset_hour):
    """
    Positive if onset is in the next six hours, matching the project's
    six-hour early-warning framing. Rows after onset are not labelled.
    """
    if pd.isna(onset_hour):
        return 0
    if hour > onset_hour:
        return None
    return int(hour <= onset_hour <= hour + PREDICTION_HOURS)


def engineered_features(window):
    """Create a tabular feature vector from a 24-hour window."""
    f = {}
    for vital in VITALS:
        col = vital + "_z"
        values = window[col].to_numpy(dtype=float)
        for n in (3, 6, 12, 24):
            x = values[-n:]
            f[f"{vital}_mean_{n}h"] = float(np.mean(x))
            f[f"{vital}_min_{n}h"] = float(np.min(x))
            f[f"{vital}_max_{n}h"] = float(np.max(x))
            f[f"{vital}_std_{n}h"] = float(np.std(x))
        f[f"{vital}_slope_6h"] = float(
            np.polyfit(np.arange(6), values[-6:], 1)[0]
        )
        f[f"{vital}_change_1h"] = float(values[-1] - values[-2])

    # Clinically interpretable derived features, using raw units.
    last = window.iloc[-1]
    hr = last["heart_rate_z"] * 1  # standardized signal retained above
    rr = last["respiratory_rate_z"]
    temp = last["temperature_z"]
    sbp = last["systolic_bp_z"]
    # SIRS/qSOFA-style flags should ideally be calculated from raw-unit
    # values before scaling. These standardized flags are not used here.
    f["mean_missing_fraction"] = float(
        window[[v + "_mask" for v in VITALS]].to_numpy().mean()
    )
    f["mean_measurement_age"] = float(
        window[[v + "_delta" for v in VITALS]].to_numpy().mean()
    )
    f["hr_z_latest"] = float(hr)
    f["rr_z_latest"] = float(rr)
    f["temp_z_latest"] = float(temp)
    f["sbp_z_latest"] = float(sbp)
    return f


def build_examples(processed):
    sequences, tabular, labels, meta = [], [], [], []
    for patient_id, group in processed.groupby("patient_id"):
        group = group.sort_values("hour").reset_index(drop=True)
        onset = group["sepsis_onset_hour"].iloc[0]
        for end in range(WINDOW_HOURS - 1, len(group)):
            window = group.iloc[end - WINDOW_HOURS + 1:end + 1]
            current_hour = float(window["hour"].iloc[-1])
            y = make_label(current_hour, onset)
            if y is None:
                continue
            seq = window[SEQ_CHANNELS].to_numpy(dtype=np.float32)
            if not np.isfinite(seq).all():
                continue
            sequences.append(seq)
            tabular.append(engineered_features(window))
            labels.append(y)
            meta.append((patient_id, current_hour))
    X_seq = np.asarray(sequences, dtype=np.float32)
    X_tab = pd.DataFrame(tabular).replace([np.inf, -np.inf], np.nan).fillna(0)
    y = np.asarray(labels, dtype=np.int64)
    return X_seq, X_tab, y, meta


# --------------------------------- LSTM -----------------------------------

class SepsisLSTM(nn.Module):
    def __init__(self, input_size=18, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        sequence_output, _ = self.lstm(x)
        weights = torch.softmax(self.attention(sequence_output), dim=1)
        context = torch.sum(weights * sequence_output, dim=1)
        x = self.dropout(context)
        x = self.relu(self.fc1(x))
        return self.output(x).squeeze(1)


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def train_lstm(X_train, y_train, X_val, y_val):
    model = SepsisLSTM(input_size=X_train.shape[2]).to(DEVICE)
    train_loader = DataLoader(
        SequenceDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        SequenceDataset(X_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False
    )

    positives = max(int(y_train.sum()), 1)
    negatives = max(len(y_train) - positives, 1)
    pos_weight = torch.tensor([negatives / positives], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=1e-5)

    best_val_loss = float("inf")
    best_state = None
    patience, wait = 5, 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_losses.append(loss_fn(model(xb), yb).item())
        val_loss = float(np.mean(val_losses))

        print(f"Epoch {epoch+1:02d}: validation loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    return model


def lstm_predict(model, X):
    model.eval()
    loader = DataLoader(
        torch.tensor(X, dtype=torch.float32),
        batch_size=BATCH_SIZE, shuffle=False
    )
    probs = []
    with torch.no_grad():
        for xb in loader:
            logits = model(xb.to(DEVICE))
            probs.extend(torch.sigmoid(logits).cpu().numpy())
    return np.asarray(probs)


# ------------------------------- Evaluation -------------------------------

def evaluate(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "sensitivity_recall": recall_score(y_true, predictions, zero_division=0),
        "specificity": tn / max(tn + fp, 1),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "auroc": roc_auc_score(y_true, probabilities),
        "auprc": average_precision_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                             "fn": int(fn), "tp": int(tp)},
    }


# --------------------------------- Main ------------------------------------

def main():
    df = pd.read_csv(CSV_PATH)
    required = {"patient_id", "hour", "sepsis_onset_hour", *VITALS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    train_raw, val_raw, test_raw = split_patients(df)

    prep = VitalPreprocessor().fit(train_raw)
    train_p = prep.transform(train_raw)
    val_p = prep.transform(val_raw)
    test_p = prep.transform(test_raw)

    Xtr_seq, Xtr_tab, ytr, _ = build_examples(train_p)
    Xva_seq, Xva_tab, yva, _ = build_examples(val_p)
    Xte_seq, Xte_tab, yte, test_meta = build_examples(test_p)

    # Logistic Regression and Random Forest are comparison baselines.
    lr = LogisticRegression(class_weight="balanced", max_iter=1000)
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        min_samples_leaf=3, random_state=SEED, n_jobs=-1
    )
    lr.fit(Xtr_tab, ytr)
    rf.fit(Xtr_tab, ytr)

    # Main tree model: gradient-boosted decision trees.
    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=max((ytr == 0).sum(), 1) /
                         max((ytr == 1).sum(), 1),
        eval_metric="aucpr", random_state=SEED,
        n_jobs=-1
    )
    xgb.fit(Xtr_tab, ytr, eval_set=[(Xva_tab, yva)], verbose=False)

    # Sequence model: two-layer LSTM.
    lstm = train_lstm(Xtr_seq, ytr, Xva_seq, yva)

    # Tune ensemble weight on validation AUPRC only.
    p_lstm_val = lstm_predict(lstm, Xva_seq)
    p_xgb_val = xgb.predict_proba(Xva_tab)[:, 1]
    best_weight, best_score = 0.6, -1
    for w in np.arange(0.0, 1.01, 0.05):
        p = w * p_lstm_val + (1 - w) * p_xgb_val
        score = average_precision_score(yva, p)
        if score > best_score:
            best_weight, best_score = float(w), float(score)

    # Fit calibration on validation predictions, never on test predictions.
    p_val_ensemble = (
        best_weight * p_lstm_val + (1 - best_weight) * p_xgb_val
    )
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(p_val_ensemble, yva)

    # Evaluate each model and the calibrated ensemble once on held-out test.
    p_lr = lr.predict_proba(Xte_tab)[:, 1]
    p_rf = rf.predict_proba(Xte_tab)[:, 1]
    p_xgb = xgb.predict_proba(Xte_tab)[:, 1]
    p_lstm = lstm_predict(lstm, Xte_seq)
    p_raw_ensemble = best_weight * p_lstm + (1 - best_weight) * p_xgb
    p_ensemble = calibrator.predict(p_raw_ensemble)

    results = {
        "Logistic Regression": evaluate(yte, p_lr),
        "Random Forest": evaluate(yte, p_rf),
        "XGBoost": evaluate(yte, p_xgb),
        "LSTM": evaluate(yte, p_lstm),
        "Calibrated LSTM + XGBoost": evaluate(yte, p_ensemble),
        "chosen_lstm_weight": best_weight,
        "chosen_xgboost_weight": 1 - best_weight,
        "test_patient_count": int(test_raw.patient_id.nunique()),
        "test_window_count": int(len(yte)),
    }
    print(json.dumps(results, indent=2))

    # Save the models and preprocessing objects for later inference.
    torch.save(lstm.state_dict(), "critical_watch_lstm.pt")
    joblib.dump(
        {"preprocessor": prep, "xgboost": xgb, "logistic": lr,
         "random_forest": rf, "calibrator": calibrator,
         "ensemble_weights": (best_weight, 1 - best_weight),
         "tabular_columns": list(Xtr_tab.columns)},
        "critical_watch_models.joblib"
    )
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
