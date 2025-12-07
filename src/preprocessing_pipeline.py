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
    """Create FIRE_SIZE_LABEL using the 'bins_04' logic from Shanti's notebook."""
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


def _undersample_to_minority(df: pd.DataFrame, random_state: int = 207) -> pd.DataFrame:
    """Undersample majority classes to match minority class size.

    This applies undersampling after train/val/test split, only to training data,
    to handle class imbalance by reducing majority classes.
    """
    # Find minimum class count
    class_counts = df["FIRE_SIZE_LABEL"].value_counts()
    min_count = class_counts.min()

    print(f"Undersampling: class counts before = {dict(class_counts)}")
    print(f"Undersampling: target count per class = {min_count}")

    # Sample min_count from each class
    df_list = []
    for label in class_counts.index:
        df_class = df[df["FIRE_SIZE_LABEL"] == label]
        if len(df_class) > min_count:
            df_class = df_class.sample(n=min_count, random_state=random_state)
        df_list.append(df_class)

    df_undersampled = pd.concat(df_list, axis=0).reset_index(drop=True)
    print(f"Undersampling: final shape = {df_undersampled.shape}")
    return df_undersampled


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
    stratify_cols: List[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reproduce the 60/20/20 split stratified by FIRE_YEAR."""
    if not np.isclose(train_pct + val_pct + test_pct, 1.0):
        raise ValueError("train_pct + val_pct + test_pct must equal 1.0")

    if stratify_cols is None:
        stratify_cols = ["FIRE_YEAR"]

    # val_size relative to (train+val) part
    val_size = val_pct / (train_pct + val_pct)

    # Determine stratify labels for the first split
    if len(stratify_cols) == 1:
        stratify_labels = df_mini[stratify_cols[0]]
    else:
        stratify_labels = list(zip(*[df_mini[c] for c in stratify_cols]))

    df_train_main, df_test = train_test_split(
        df_mini,
        test_size=test_pct,
        random_state=random_state,
        stratify=stratify_labels,
    )

    # Determine stratify labels for the second split
    if len(stratify_cols) == 1:
        stratify_labels_main = df_train_main[stratify_cols[0]]
    else:
        stratify_labels_main = list(zip(*[df_train_main[c] for c in stratify_cols]))

    df_train, df_val = train_test_split(
        df_train_main,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_labels_main,
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


def _integer_encode_categoricals(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    categorical_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[str, Any]]:
    """
    Integer-encode categorical columns for entity embeddings.

    Instead of one-hot encoding, converts categorical values to integer indices
    that can be used with Keras Embedding layers.

    Args:
        df_train, df_val, df_test: DataFrames to transform
        categorical_cols: List of column names to encode

    Returns:
        Tuple of (df_train, df_val, df_test, cardinalities, encoders)
        - cardinalities: Dict mapping column name to number of unique values
        - encoders: Dict mapping column name to LabelEncoder instance
    """
    from sklearn.preprocessing import LabelEncoder

    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()

    cardinalities = {}
    encoders = {}

    for col in categorical_cols:
        if col not in df_train.columns:
            continue

        # Fit encoder on all unique values across splits to handle unseen categories
        le = LabelEncoder()
        all_values = pd.concat([
            df_train[col].astype(str),
            df_val[col].astype(str),
            df_test[col].astype(str)
        ]).unique()
        le.fit(all_values)

        # Transform each split
        df_train[col] = le.transform(df_train[col].astype(str))
        df_val[col] = le.transform(df_val[col].astype(str))
        df_test[col] = le.transform(df_test[col].astype(str))

        cardinalities[col] = len(le.classes_)
        encoders[col] = le

        print(f"  {col}: {cardinalities[col]} unique values")

    return df_train, df_val, df_test, cardinalities, encoders


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
    use_undersample: bool = False,
    use_entity_embeddings: bool = False,
) -> Dict[str, Any]:
    """
    End-to-end preprocessing pipeline that mirrors Shanti's notebook, starting from raw CSV.

    Args:
        n_per_class: Number of samples per class for balanced sampling (default: 1000).
                     Set to None to skip balanced sampling and use full dataset.
        random_state: Random seed for reproducibility (default: 207)
        use_smote: If True, use SMOTE oversampling on training data (default: False)
        use_undersample: If True, undersample majority classes to minority size (default: False)
        use_entity_embeddings: If True, return integer-encoded categoricals for entity embeddings
                               instead of one-hot encoding. Returns separate continuous/categorical
                               arrays with cardinality metadata. (default: False)

    Returns a dict containing:
      - X_train_cont_std, X_val_cont_std, X_test_cont_std (scaled continuous features)
      - X_train_std, X_val_std, X_test_std (concatenated features - only when not using entity embeddings)
      - Y_train, Y_val, Y_test
      - cols_continuous, cols_onehot, feature_cols
      - df_train_processed, df_val_processed, df_test_processed

      When use_entity_embeddings=True, also includes:
      - X_train_cat, X_val_cat, X_test_cat (integer-encoded categorical features)
      - cardinalities: Dict mapping categorical column to number of unique values
      - cols_categorical: List of categorical column names

    Note: use_smote and use_undersample are mutually exclusive. If both are True, use_smote takes precedence.
    """
    # 1. Load raw data
    df_init = load_data_from_csv(convert_to_pandas=True)
    if df_init is None:
        raise RuntimeError("Failed to load data from CSV via data_loader.load_data_from_csv.")

    # 2. Drop columns with >80% missing
    df = _drop_high_missing_columns(df_init)

    # 3. Create FIRE_SIZE_LABEL
    df = _create_fire_size_label(df)

    # 4. Balanced sampling (skip if using SMOTE, undersampling, or n_per_class=None)
    if use_smote or use_undersample or n_per_class is None:
        df_mini = df
        if n_per_class is None:
            print("Using full dataset (n_per_class=None)")
    else:
        df_mini = _sample_balanced_classes(df, n_per_class=n_per_class, random_state=random_state)

    # 5. Map FIRE_SIZE_LABEL to numeric
    df_mini = _encode_fire_size_label_numeric(df_mini)

    # 6. Shuffle rows
    rng = np.random.RandomState(random_state)
    shuffled_idx = rng.permutation(df_mini.index)
    df_mini = df_mini.loc[shuffled_idx].reset_index(drop=True)

    # 7. Train/val/test split (60/20/20) stratified by FIRE_YEAR
    # If using SMOTE, undersampling, or full dataset, also stratify by FIRE_SIZE_LABEL to preserve class distribution
    stratify_cols = ["FIRE_YEAR"]
    if use_smote or use_undersample or n_per_class is None:
        stratify_cols.append("FIRE_SIZE_LABEL")

    df_train, df_val, df_test = _train_val_test_split(
        df_mini,
        random_state=random_state,
        stratify_cols=stratify_cols
    )

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

    # 12. Encode categorical features
    CATEGORICAL_COLS = ['NWCG_CAUSE_CLASSIFICATION', 'GACC_PL']

    if use_entity_embeddings:
        # Integer encode for entity embeddings
        print("Integer-encoding categorical features for entity embeddings...")
        df_train, df_val, df_test, cardinalities, encoders = _integer_encode_categoricals(
            df_train, df_val, df_test, CATEGORICAL_COLS
        )
        cols_categorical = [c for c in CATEGORICAL_COLS if c in df_train.columns]
    else:
        # One-hot encode (existing behavior)
        df_train, df_val, df_test = _one_hot_encode(df_train, df_val, df_test)
        cardinalities = None
        cols_categorical = []

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

    elif use_undersample:
        # Apply undersampling to training data only
        print("Applying undersampling to training data...")
        df_train = _undersample_to_minority(df_train, random_state=random_state)
        print("Undersampling done.")

    # 13. Standardize continuous features and return arrays/metadata
    if use_entity_embeddings:
        # Entity embeddings path: separate continuous and categorical
        y_train = df_train["FIRE_SIZE_LABEL"].values
        y_val = df_val["FIRE_SIZE_LABEL"].values
        y_test = df_test["FIRE_SIZE_LABEL"].values

        # Continuous columns (exclude target and categorical)
        cols_continuous = [c for c in df_train.columns
                          if c not in ['FIRE_SIZE_LABEL'] + cols_categorical]

        # Scale continuous features
        scaler = StandardScaler()
        X_train_cont = scaler.fit_transform(df_train[cols_continuous])
        X_val_cont = scaler.transform(df_val[cols_continuous])
        X_test_cont = scaler.transform(df_test[cols_continuous])

        # Extract categorical as integers
        X_train_cat = df_train[cols_categorical].values.astype('int32')
        X_val_cat = df_val[cols_categorical].values.astype('int32')
        X_test_cat = df_test[cols_categorical].values.astype('int32')

        result = {
            # Continuous (scaled)
            'X_train_cont': X_train_cont,
            'X_val_cont': X_val_cont,
            'X_test_cont': X_test_cont,
            # Also include with _std suffix for compatibility
            'X_train_cont_std': X_train_cont,
            'X_val_cont_std': X_val_cont,
            'X_test_cont_std': X_test_cont,
            # Categorical (integer indices)
            'X_train_cat': X_train_cat,
            'X_val_cat': X_val_cat,
            'X_test_cat': X_test_cat,
            # Targets
            'Y_train': y_train,
            'Y_val': y_val,
            'Y_test': y_test,
            # Metadata for building embedding model
            'cardinalities': cardinalities,
            'cols_continuous': cols_continuous,
            'cols_categorical': cols_categorical,
            'feature_cols': cols_continuous + cols_categorical,
            # Keep one-hot related keys empty for compatibility
            'cols_onehot': [],
        }
    else:
        # Original one-hot path
        result = _standardize_features(df_train, df_val, df_test)

    # keep the processed dataframes (after encoding/imputation) if you want to inspect later
    result["df_train_processed"] = df_train
    result["df_val_processed"] = df_val
    result["df_test_processed"] = df_test

    return result


if __name__ == "__main__":
    # check output
    data = build_preprocessed_data(use_smote=True)
    Y_train_resampled = data["Y_train"]
    X_train_res_std = data["X_train_std"]
    Y_val = data["Y_val"]
    X_val_std = data["X_val_std"]
    Y_test = data["Y_test"]
    X_test_std = data["X_test_std"]
    pairs = [(Y_train_resampled, X_train_res_std, 'Training Data'),
             (Y_val, X_val_std, 'Validation Data'),
             (Y_test, X_test_std, 'Test Data')]

    for pair in pairs:
        print(f"{'-' * 10} {pair[2]} {'-' * 10}")
        print(f"Shape of Y: {pair[0].shape}; Shape of X: {pair[1].shape}")
        print(f"Y: {np.mean(pair[0]): .7f}")
        print(f"X Mean: {np.mean(pair[1]): .7f}")
        print(f"X Std: {np.std(pair[1]): .7f}")
        print(f"X Min: {np.min(pair[1]): .7f}")
        print(f"X Max: {np.min(pair[1]): .7f}\n")

    # feature mean
    for pair in pairs:
        print(f"{'-' * 10} {pair[2]} : Feature Mean {'-' * 10}")
        print(f"Number of features: {len(np.mean(pair[1], axis=0))}")  # sanity check
        print(f"{np.mean(pair[1], axis=0)}\n")
