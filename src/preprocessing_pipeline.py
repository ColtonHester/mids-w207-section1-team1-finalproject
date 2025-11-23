from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_data_from_csv


def _drop_high_missing_columns(df: pd.DataFrame, thresh_ratio: float = 0.8) -> pd.DataFrame:
    """Drop columns whose number of missing values is greater than thresh_ratio * n_rows."""
    cutoff_missing_values = df.shape[0] * thresh_ratio
    missing_vals = df.isna().sum().sort_values(ascending=False)
    cols_missing_gt_thresh = list(missing_vals[missing_vals > cutoff_missing_values].index)
    return df.drop(columns=cols_missing_gt_thresh)


def _create_fire_size_label(df: pd.DataFrame) -> pd.DataFrame:
    """Create FIRE_SIZE_LABEL using the 'bins_04' logic from 's notebook."""
    bins_04 = [0, 100, 4999, 29000, df["FIRE_SIZE"].max()]
    group_names = ["small", "medium", "large", "very large"]
    df = df.copy()
    df["FIRE_SIZE_LABEL"] = pd.cut(df["FIRE_SIZE"], bins_04, labels=group_names)
    return df


def _sample_balanced_classes(df: pd.DataFrame, n_per_class: int = 1000, random_state: int = 207) -> pd.DataFrame:
    """Sample n_per_class rows from each FIRE_SIZE_LABEL class using FPA_ID-based sampling."""
    rng = np.random.RandomState(random_state)

    # ensure we have the column
    if "FPA_ID" not in df.columns:
        raise ValueError("Expected column 'FPA_ID' not found in dataframe.")

    labels = ["small", "medium", "large", "very large"]
    df_list = []

    for label in labels:
        fpaid = df.loc[df["FIRE_SIZE_LABEL"] == label, "FPA_ID"].values
        if len(fpaid) < n_per_class:
            raise ValueError(
                f"Not enough samples in class '{label}' to draw {n_per_class} without replacement "
                f"(available: {len(fpaid)})"
            )

        sampled_ids = rng.choice(fpaid, size=n_per_class, replace=False)
        df_label = df[df["FPA_ID"].isin(sampled_ids)]
        df_list.append(df_label)

    df_mini = pd.concat(df_list, axis=0).reset_index(drop=True)
    return df_mini


def _encode_fire_size_label_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Map FIRE_SIZE_LABEL from string to numeric 0..3."""
    mapping = {
        "small": 0,
        "medium": 1,
        "large": 2,
        "very large": 3,
    }
    df = df.copy()
    df["FIRE_SIZE_LABEL"] = df["FIRE_SIZE_LABEL"].map(mapping)
    return df


def _train_val_test_split(
    df_mini: pd.DataFrame,
    train_pct: float = 0.6,
    val_pct: float = 0.2,
    test_pct: float = 0.2,
    random_state: int = 207,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reproduce the 60/20/20 split stratified by FIRE_YEAR."""
    if not np.isclose(train_pct + val_pct + test_pct, 1.0):
        raise ValueError("train_pct + val_pct + test_pct must equal 1.0")

    # val_size relative to (train+val) part
    val_size = val_pct / (train_pct + val_pct)

    df_train_main, df_test = train_test_split(
        df_mini,
        test_size=test_pct,
        random_state=random_state,
        stratify=df_mini["FIRE_YEAR"],
    )

    df_train, df_val = train_test_split(
        df_train_main,
        test_size=val_size,
        random_state=random_state,
        stratify=df_train_main["FIRE_YEAR"],
    )

    return df_train, df_val, df_test


def _select_features(df_train: pd.DataFrame) -> List[str]:
    """Construct the feature list - Same as in Shanti's notebook."""
    # from FPA FOD data
    trgt_feat_fod = [
        "FIRE_SIZE_LABEL",
        "FIRE_YEAR",
        "DISCOVERY_DOY",
        "NWCG_CAUSE_CLASSIFICATION",
        "LATITUDE",
        "LONGITUDE",
    ]

    # from GRIDMET: variables containing '_5D_'
    feat_gridmet = list(df_train.columns[df_train.columns.str.contains("_5D_")])

    # from risk management assistance
    feat_rmgmta = ["SDI"]

    # from fire stations
    feat_firestation = ["No_FireStation_20.0km"]

    # from GACC
    feat_gacc = ["GACC_PL"]

    # from global human modification
    feat_ghm = ["GHM"]

    # from NDVI
    feat_ndvi = ["NDVI-1day"]

    # from national preparedness level
    feat_npl = ["NPL"]

    # from social vulnerability index
    feat_svi = ["EPL_PCI"]

    # from rangeland production
    feat_rangeland = ["rpms", "rpms_1km"]

    trgt_feat_selected = (
        trgt_feat_fod
        + feat_gridmet
        + feat_rmgmta
        + feat_firestation
        + feat_gacc
        + feat_ghm
        + feat_ndvi
        + feat_npl
        + feat_svi
        + feat_rangeland
    )

    # Filter to those that actually exist in df_train
    trgt_feat_selected = [c for c in trgt_feat_selected if c in df_train.columns]
    return trgt_feat_selected


def _impute_with_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Impute the missing values in selected columns with zero."""
    df = df.copy()
    cols_impute_zero = ["No_FireStation_20.0km", "GACC_PL"]
    for col in cols_impute_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df


def _impute_with_mean(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feat_gridmet: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Impute GRIDMET + SDI + EPL_PCI + NDVI-1day with training mean."""
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()

    # Replace -999 in EPL_PCI with NaN
    for data in [df_train, df_val, df_test]:
        if "EPL_PCI" in data.columns:
            data["EPL_PCI"] = data["EPL_PCI"].replace(-999, np.nan)

    cols_impute_mean = feat_gridmet + ["SDI", "EPL_PCI", "NDVI-1day"]
    cols_impute_mean = [c for c in cols_impute_mean if c in df_train.columns]

    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    df_train[cols_impute_mean] = imp.fit_transform(df_train[cols_impute_mean])
    df_val[cols_impute_mean] = imp.transform(df_val[cols_impute_mean])
    df_test[cols_impute_mean] = imp.transform(df_test[cols_impute_mean])

    return df_train, df_val, df_test


def _clean_nwcg_cause_classification(df_list: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Apply mapping to NWCG_CAUSE_CLASSIFICATION."""
    mapping = {
        "Human": "hum",
        "Natural": "nat",
        "Missing data/not specified/undetermined": "miss-unspec",
    }

    cleaned = []
    for df in df_list:
        df = df.copy()
        if "NWCG_CAUSE_CLASSIFICATION" in df.columns:
            df["NWCG_CAUSE_CLASSIFICATION"] = df["NWCG_CAUSE_CLASSIFICATION"].map(mapping)
        cleaned.append(df)
    return cleaned


def _one_hot_encode(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
    """One-hot encode NWCG_CAUSE_CLASSIFICATION and GACC_PL."""
    cols_onehot = []
    for col in ["NWCG_CAUSE_CLASSIFICATION", "GACC_PL"]:
        if col in df_train.columns:
            cols_onehot.append(col)

    df_train = pd.get_dummies(df_train, columns=cols_onehot)
    df_val = pd.get_dummies(df_val, columns=cols_onehot)
    df_test = pd.get_dummies(df_test, columns=cols_onehot)

    # Align columns across splits
    df_train, df_val = df_train.align(df_val, join="left", axis=1, fill_value=0)
    df_train, df_test = df_train.align(df_test, join="left", axis=1, fill_value=0)

    return df_train, df_val, df_test


def _standardize_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Dict[str, Any]:
    """Standardize continuous features, keep one-hot as is, return arrays and metadata."""
    # outcome
    y_train = df_train["FIRE_SIZE_LABEL"].values
    y_val = df_val["FIRE_SIZE_LABEL"].values
    y_test = df_test["FIRE_SIZE_LABEL"].values

    # feature set (drop outcome)
    feature_cols = [c for c in df_train.columns if c != "FIRE_SIZE_LABEL"]

    X_train = df_train[feature_cols]
    X_val = df_val[feature_cols]
    X_test = df_test[feature_cols]

    # identify one-hot columns
    cols_nwcg = [c for c in X_train.columns if "NWCG_CAUSE_CLASSIFICATION" in c]
    cols_gacc = [c for c in X_train.columns if "GACC_PL" in c]
    cols_onehot = cols_nwcg + cols_gacc

    cols_continuous = [c for c in X_train.columns if c not in cols_onehot]

    X_train_continuous = X_train[cols_continuous]
    X_val_continuous = X_val[cols_continuous]
    X_test_continuous = X_test[cols_continuous]

    X_train_onehot = X_train[cols_onehot] if cols_onehot else pd.DataFrame(index=X_train.index)
    X_val_onehot = X_val[cols_onehot] if cols_onehot else pd.DataFrame(index=X_val.index)
    X_test_onehot = X_test[cols_onehot] if cols_onehot else pd.DataFrame(index=X_test.index)

    scaler = StandardScaler()
    X_train_cont_std = scaler.fit_transform(X_train_continuous)
    X_val_cont_std = scaler.transform(X_val_continuous)
    X_test_cont_std = scaler.transform(X_test_continuous)

    # concatenate continuous + one-hot
    if cols_onehot:
        X_train_std = np.concatenate([X_train_cont_std, X_train_onehot.values], axis=1)
        X_val_std = np.concatenate([X_val_cont_std, X_val_onehot.values], axis=1)
        X_test_std = np.concatenate([X_test_cont_std, X_test_onehot.values], axis=1)
    else:
        X_train_std = X_train_cont_std
        X_val_std = X_val_cont_std
        X_test_std = X_test_cont_std

    return {
        "X_train_cont_std": X_train_cont_std,
        "X_val_cont_std": X_val_cont_std,
        "X_test_cont_std": X_test_cont_std,
        "X_train_std": X_train_std,
        "X_val_std": X_val_std,
        "X_test_std": X_test_std,
        "Y_train": y_train,
        "Y_val": y_val,
        "Y_test": y_test,
        "cols_continuous": cols_continuous,
        "cols_onehot": cols_onehot,
        "feature_cols": feature_cols,
    }


def build_preprocessed_data(
    n_per_class: int = 1000,
    random_state: int = 207,
    use_smote: bool = False,
) -> Dict[str, Any]:
    """
    End-to-end preprocessing pipeline that mirrors Shanti's notebook, starting from raw CSV.

    Returns a dict containing:
      - X_train_cont_std, X_val_cont_std, X_test_cont_std
      - X_train_std, X_val_std, X_test_std
      - Y_train, Y_val, Y_test
      - cols_continuous, cols_onehot, feature_cols
      - df_train_raw, df_val_raw, df_test_raw (before standardization, after encoding/imputation)
    """
    # 1. Load raw data
    df_init = load_data_from_csv(convert_to_pandas=True)
    if df_init is None:
        raise RuntimeError("Failed to load data from CSV via data_loader.load_data_from_csv.")

    # 2. Drop columns with >80% missing
    df = _drop_high_missing_columns(df_init)

    # 3. Create FIRE_SIZE_LABEL
    df = _create_fire_size_label(df)

    # 4. Balanced sampling
    if use_smote:
        df_mini = df
    else:
        df_mini = _sample_balanced_classes(df, n_per_class=n_per_class, random_state=random_state)

    # 5. Map FIRE_SIZE_LABEL to numeric
    df_mini = _encode_fire_size_label_numeric(df_mini)

    # 6. Shuffle rows
    rng = np.random.RandomState(random_state)
    shuffled_idx = rng.permutation(df_mini.index)
    df_mini = df_mini.loc[shuffled_idx].reset_index(drop=True)

    # 7. Train/val/test split (60/20/20) stratified by FIRE_YEAR
    df_train, df_val, df_test = _train_val_test_split(df_mini, random_state=random_state)

    # 8. Select features of interest (same logic as notebook)
    trgt_feat_selected = _select_features(df_train)
    df_train = df_train[trgt_feat_selected].copy()
    df_val = df_val[trgt_feat_selected].copy()
    df_test = df_test[trgt_feat_selected].copy()

    # 9. Impute selected vars with zero
    df_train = _impute_with_zero(df_train)
    df_val = _impute_with_zero(df_val)
    df_test = _impute_with_zero(df_test)

    # 10. Impute mean for GRIDMET + SDI + EPL_PCI + NDVI-1day
    feat_gridmet = [c for c in df_train.columns if "_5D_" in c]
    df_train, df_val, df_test = _impute_with_mean(df_train, df_val, df_test, feat_gridmet)

    # 11. Clean NWCG_CAUSE_CLASSIFICATION
    df_train, df_val, df_test = _clean_nwcg_cause_classification([df_train, df_val, df_test])

    # 12. One-hot encode NWCG_CAUSE_CLASSIFICATION and GACC_PL
    df_train, df_val, df_test = _one_hot_encode(df_train, df_val, df_test)

    if use_smote:
        X_train = df_train.drop(columns=["FIRE_SIZE_LABEL"])
        y_train = df_train["FIRE_SIZE_LABEL"]

        # Apply SMOTE
        print("Applying SMOTE...")
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print("SMOTE done.")

        # Reconstruct df_train with resampled data
        df_train = pd.concat([X_train_res, y_train_res], axis=1)

    # 13. Standardize continuous features and return arrays/metadata
    result = _standardize_features(df_train, df_val, df_test)

    # keep the processed dataframes (after encoding/imputation) if you want to inspect later
    result["df_train_processed"] = df_train
    result["df_val_processed"] = df_val
    result["df_test_processed"] = df_test

    return result


if __name__ == "__main__":
    # check output
    data = build_preprocessed_data()
    print("X_train_cont_std shape:", data["X_train_cont_std"].shape)
    print("X_train_std shape:", data["X_train_std"].shape)
    print("Y_train shape:", data["Y_train"].shape)
    print("Number of continuous features:", len(data["cols_continuous"]))
    print("Number of one-hot features:", len(data["cols_onehot"]))