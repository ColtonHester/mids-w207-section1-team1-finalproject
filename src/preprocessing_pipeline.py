from __future__ import annotations

import numpy as np
import pandas as pd
import gc
from typing import Dict, Any, List, Tuple, Optional
from imblearn.over_sampling import SMOTE, SMOTENC

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.data_loader import load_data_from_csv


# Adding some memory checks to monitor memory usage during preprocessing
def _get_memory_usage_gb() -> float:
    """Return current process memory usage in GB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except ImportError:
        return -1.0


def _log_memory(stage: str) -> None:
    """Log memory usage at a given stage."""
    mem = _get_memory_usage_gb()
    if mem > 0:
        print(f"[Memory] {stage}: {mem:.2f} GB")


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
    df["FIRE_SIZE_LABEL"] = pd.cut(df["FIRE_SIZE"], bins_04, labels=group_names)
    return df


def _sample_balanced_classes(df: pd.DataFrame, n_per_class: int = 1000, random_state: int = 207) -> pd.DataFrame:
    """Sample n_per_class rows from each FIRE_SIZE_LABEL class using FPA_ID-based sampling."""
    rng = np.random.RandomState(random_state)

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


def _sample_stratified_for_smote(
        df: pd.DataFrame,
        minority_n: int = 2000,
        majority_cap: int = 8000,
        random_state: int = 207
) -> pd.DataFrame:
    """
    Sample data for SMOTE: keep more minority samples, cap majority class.

    This creates an imbalanced but manageable dataset that SMOTE can then balance.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with FIRE_SIZE_LABEL column
    minority_n : int
        Target samples for minority classes (or all available if fewer exist)
    majority_cap : int
        Maximum samples from majority class
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Stratified sample suitable for SMOTE augmentation
    """
    rng = np.random.RandomState(random_state)

    if "FPA_ID" not in df.columns:
        raise ValueError("Expected column 'FPA_ID' not found in dataframe.")

    # Get class distribution
    class_counts = df["FIRE_SIZE_LABEL"].value_counts()
    print(f"Original class distribution:\n{class_counts}")

    labels = ["small", "medium", "large", "very large"]
    df_list = []

    for label in labels:
        fpaid = df.loc[df["FIRE_SIZE_LABEL"] == label, "FPA_ID"].values
        available = len(fpaid)

        # Determine sample size: minority_n for small classes, capped for large
        if available <= minority_n:
            # Take all available for rare classes
            sample_size = available
        elif label == "small":  # Typically the majority class
            sample_size = min(available, majority_cap)
        else:
            sample_size = min(available, minority_n)

        sampled_ids = rng.choice(fpaid, size=sample_size, replace=False)
        df_label = df[df["FPA_ID"].isin(sampled_ids)]
        df_list.append(df_label)
        print(f"  {label}: sampled {sample_size} from {available}")

    df_sampled = pd.concat(df_list, axis=0).reset_index(drop=True)
    print(f"Total sampled: {len(df_sampled)} rows")
    return df_sampled


def _encode_fire_size_label_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Map FIRE_SIZE_LABEL from string to numeric 0..3."""
    mapping = {
        "small": 0,
        "medium": 1,
        "large": 2,
        "very large": 3,
    }
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

    val_size = val_pct / (train_pct + val_pct)

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
    trgt_feat_fod = [
        "FIRE_SIZE_LABEL",
        "FIRE_YEAR",
        "DISCOVERY_DOY",
        "NWCG_CAUSE_CLASSIFICATION",
        "LATITUDE",
        "LONGITUDE",
    ]

    feat_gridmet = list(df_train.columns[df_train.columns.str.contains("_5D_")])
    feat_rmgmta = ["SDI"]
    feat_firestation = ["No_FireStation_20.0km"]
    feat_gacc = ["GACC_PL"]
    feat_ghm = ["GHM"]
    feat_ndvi = ["NDVI-1day"]
    feat_npl = ["NPL"]
    feat_svi = ["EPL_PCI"]
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

    trgt_feat_selected = [c for c in trgt_feat_selected if c in df_train.columns]
    return trgt_feat_selected


def _impute_with_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Impute the missing values in selected columns with zero."""
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


def _impute_with_subgroup_mean(
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        feat_gridmet: List[str],
        subgroup_col: str = "GACC_PL",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Impute GRIDMET + SDI + EPL_PCI + NDVI-1day with subgroup mean (e.g. by GACC_PL).
    Falls back to global mean if subgroup mean is NaN.
    """

    for data in [df_train, df_val, df_test]:
        if "EPL_PCI" in data.columns:
            data["EPL_PCI"] = data["EPL_PCI"].replace(-999, np.nan)

    cols_impute = feat_gridmet + ["SDI", "EPL_PCI", "NDVI-1day"]
    cols_impute = [c for c in cols_impute if c in df_train.columns]

    if subgroup_col not in df_train.columns:
        print(f"Warning: Subgroup column '{subgroup_col}' not found. Falling back to global mean imputation.")
        return _impute_with_mean(df_train, df_val, df_test, feat_gridmet)

    subgroup_means = df_train.groupby(subgroup_col)[cols_impute].mean()
    global_means = df_train[cols_impute].mean()

    def impute_dataframe(df_target, means_lookup, global_fallback):
        for col in cols_impute:
            mapped_means = df_target[subgroup_col].map(means_lookup[col])
            df_target[col] = df_target[col].fillna(mapped_means)
            if df_target[col].isna().any():
                df_target[col] = df_target[col].fillna(global_fallback[col])
        return df_target

    df_train = impute_dataframe(df_train, subgroup_means, global_means)
    df_val = impute_dataframe(df_val, subgroup_means, global_means)
    df_test = impute_dataframe(df_test, subgroup_means, global_means)

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
    train_cols = df_train.columns

    def align_to_train(df, target_cols):
        missing = target_cols.difference(df.columns)
        if not missing.empty:
            df_zeros = pd.DataFrame(0, index=df.index, columns=missing)
            df = pd.concat([df, df_zeros], axis=1)

        extra = df.columns.difference(target_cols)
        if not extra.empty:
            df.drop(columns=extra, inplace=True)

        return df[target_cols]

    df_val = align_to_train(df_val, train_cols)
    df_test = align_to_train(df_test, train_cols)

    return df_train, df_val, df_test


def _label_encode_categoricals(
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        cat_cols: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Label encode categorical columns for embedding models.

    Returns the dataframes with label-encoded categoricals and a dict of fitted encoders.
    Unknown categories in val/test are mapped to a special index (num_classes).
    """
    if cat_cols is None:
        cat_cols = ["NWCG_CAUSE_CLASSIFICATION", "GACC_PL"]

    cat_cols = [c for c in cat_cols if c in df_train.columns]
    label_encoders = {}

    for col in cat_cols:
        le = LabelEncoder()

        # Fill NaN with a placeholder before encoding
        df_train[col] = df_train[col].fillna("_MISSING_")
        df_val[col] = df_val[col].fillna("_MISSING_")
        df_test[col] = df_test[col].fillna("_MISSING_")

        # Fit on training data
        le.fit(df_train[col].astype(str))
        label_encoders[col] = le

        # Transform training data
        df_train[col] = le.transform(df_train[col].astype(str))

        # Handle unknown categories in val/test by mapping to num_classes (OOV index)
        def safe_transform(series, encoder):
            known_classes = set(encoder.classes_)
            oov_idx = len(encoder.classes_)  # Index for out-of-vocabulary
            return series.astype(str).apply(
                lambda x: encoder.transform([x])[0] if x in known_classes else oov_idx
            )

        df_val[col] = safe_transform(df_val[col], le)
        df_test[col] = safe_transform(df_test[col], le)

    return df_train, df_val, df_test, label_encoders


def _apply_smotenc(
        df_train: pd.DataFrame,
        random_state: int = 207,
        sampling_strategy: str = "auto"
) -> pd.DataFrame:
    """
    Apply SMOTENC to training data with memory-safe practices.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data with FIRE_SIZE_LABEL as target
    random_state : int
        Random seed
    sampling_strategy : str or dict
        SMOTE sampling strategy. Use 'auto' to balance all classes,
        or a dict like {1: 5000, 2: 5000, 3: 5000} to specify target counts.

    Returns
    -------
    pd.DataFrame
        Resampled training data
    """
    _log_memory("Before SMOTENC")

    X_train = df_train.drop(columns=["FIRE_SIZE_LABEL"])
    y_train = df_train["FIRE_SIZE_LABEL"]
    x_columns = X_train.columns.tolist()

    print(f"SMOTENC input shape: {X_train.shape}")
    print(f"Class distribution before SMOTE:\n{y_train.value_counts().sort_index()}")

    # Identify categorical features
    cat_cols = ["NWCG_CAUSE_CLASSIFICATION", "GACC_PL", "STATE"]
    present_cat_cols = [c for c in cat_cols if c in X_train.columns]

    if not present_cat_cols:
        raise ValueError("No categorical columns found for SMOTENC. Use standard SMOTE instead.")

    # Label encode categoricals (suitable for SMOTENC)
    label_encoders = {}
    for c in present_cat_cols:
        X_train[c] = X_train[c].fillna("Missing")
        le = LabelEncoder()
        X_train[c] = le.fit_transform(X_train[c].astype(str))
        label_encoders[c] = le

    # Handle inf/nan in continuous features
    continuous_cols = [c for c in X_train.columns if c not in present_cat_cols]
    if continuous_cols:
        X_cont = X_train[continuous_cols].astype("float64")
        inf_mask = ~np.isfinite(X_cont)
        if inf_mask.values.any():
            print("Warning: Replacing inf/-inf values with column means.")
            X_cont[inf_mask] = np.nan
            col_means = X_cont.mean(axis=0)
            X_cont = X_cont.fillna(col_means)
            for col in continuous_cols:
                orig_dtype = X_train[col].dtype
                if np.issubdtype(orig_dtype, np.number):
                    X_train[col] = X_cont[col].astype(orig_dtype)
                else:
                    X_train[col] = X_cont[col]

    cat_indices = [X_train.columns.get_loc(c) for c in present_cat_cols]

    # Apply SMOTENC
    smotenc = SMOTENC(
        categorical_features=cat_indices,
        random_state=random_state,
        sampling_strategy=sampling_strategy,
        # n_jobs=-1  # Use all cores for k-NN
    )

    _log_memory("Before fit_resample")
    X_train_res, y_train_res = smotenc.fit_resample(X_train, y_train)
    _log_memory("After fit_resample")

    print(f"SMOTENC output shape: {X_train_res.shape}")
    print(f"Class distribution after SMOTE:\n{pd.Series(y_train_res).value_counts().sort_index()}")

    # Cleanup
    del X_train, y_train, smotenc
    gc.collect()

    # Convert back to DataFrame
    if not isinstance(X_train_res, pd.DataFrame):
        X_train_res = pd.DataFrame(X_train_res, columns=x_columns)

    # Inverse transform categoricals
    for c in present_cat_cols:
        le = label_encoders[c]
        X_train_res[c] = le.inverse_transform(X_train_res[c].astype(int))

    # Reconstruct full dataframe
    df_train_resampled = pd.concat([X_train_res, pd.Series(y_train_res, name="FIRE_SIZE_LABEL")], axis=1)

    del X_train_res, y_train_res
    gc.collect()

    return df_train_resampled


def _apply_smote(
        df_train: pd.DataFrame,
        random_state: int = 207,
        sampling_strategy: str = "auto"
) -> pd.DataFrame:
    """Apply standard SMOTE to training data (after one-hot encoding)."""
    _log_memory("Before SMOTE")

    X_train = df_train.drop(columns=["FIRE_SIZE_LABEL"])
    y_train = df_train["FIRE_SIZE_LABEL"]

    print(f"SMOTE input shape: {X_train.shape}")
    print(f"Class distribution before SMOTE:\n{y_train.value_counts().sort_index()}")

    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy, n_jobs=-1)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"SMOTE output shape: {X_train_res.shape}")
    print(f"Class distribution after SMOTE:\n{pd.Series(y_train_res).value_counts().sort_index()}")

    _log_memory("After SMOTE")

    df_train_resampled = pd.concat([
        pd.DataFrame(X_train_res, columns=X_train.columns),
        pd.Series(y_train_res, name="FIRE_SIZE_LABEL")
    ], axis=1)

    return df_train_resampled


def _standardize_features(
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
) -> Dict[str, Any]:
    """Standardize continuous features, keep one-hot as is, return arrays and metadata."""
    y_train = df_train["FIRE_SIZE_LABEL"].values
    y_val = df_val["FIRE_SIZE_LABEL"].values
    y_test = df_test["FIRE_SIZE_LABEL"].values

    feature_cols = [c for c in df_train.columns if c != "FIRE_SIZE_LABEL"]

    X_train = df_train[feature_cols]
    X_val = df_val[feature_cols]
    X_test = df_test[feature_cols]

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
        "scaler": scaler,
    }


def _standardize_features_for_embeddings(
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        cat_cols: List[str],
        label_encoders: Dict[str, LabelEncoder],
) -> Dict[str, Any]:
    """
    Prepare features for embedding models:
    - Categorical features as integer arrays (for embedding lookup)
    - Continuous features standardized
    - Returns both separately for multi-input model architecture
    """
    y_train = df_train["FIRE_SIZE_LABEL"].values
    y_val = df_val["FIRE_SIZE_LABEL"].values
    y_test = df_test["FIRE_SIZE_LABEL"].values

    feature_cols = [c for c in df_train.columns if c != "FIRE_SIZE_LABEL"]

    # Separate categorical and continuous columns
    present_cat_cols = [c for c in cat_cols if c in feature_cols]
    cols_continuous = [c for c in feature_cols if c not in present_cat_cols]

    # Extract categorical features as integers
    X_train_cat = df_train[present_cat_cols].values.astype('int32') if present_cat_cols else None
    X_val_cat = df_val[present_cat_cols].values.astype('int32') if present_cat_cols else None
    X_test_cat = df_test[present_cat_cols].values.astype('int32') if present_cat_cols else None

    # Extract and standardize continuous features
    X_train_continuous = df_train[cols_continuous].astype('float32')
    X_val_continuous = df_val[cols_continuous].astype('float32')
    X_test_continuous = df_test[cols_continuous].astype('float32')

    scaler = StandardScaler()
    X_train_cont_std = scaler.fit_transform(X_train_continuous).astype('float32')
    X_val_cont_std = scaler.transform(X_val_continuous).astype('float32')
    X_test_cont_std = scaler.transform(X_test_continuous).astype('float32')

    # Build categorical feature info for model construction
    categorical_info = {}
    for col in present_cat_cols:
        le = label_encoders[col]
        num_categories = len(le.classes_) + 1  # +1 for OOV

        embed_dim = min(50, (num_categories + 1) // 2)
        embed_dim = max(embed_dim, 2)  # At least 2 dimensions
        categorical_info[col] = {
            'num_categories': num_categories,
            'embed_dim': embed_dim,
            'col_idx': present_cat_cols.index(col),
        }

    return {
        # Categorical features (integer encoded for embeddings)
        "X_train_cat": X_train_cat,
        "X_val_cat": X_val_cat,
        "X_test_cat": X_test_cat,
        # Continuous features (standardized)
        "X_train_cont": X_train_cont_std,
        "X_val_cont": X_val_cont_std,
        "X_test_cont": X_test_cont_std,
        # Labels
        "Y_train": y_train,
        "Y_val": y_val,
        "Y_test": y_test,
        # Metadata
        "cols_categorical": present_cat_cols,
        "cols_continuous": cols_continuous,
        "categorical_info": categorical_info,
        "label_encoders": label_encoders,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }


def build_preprocessed_data(
        n_per_class: int = 1000,
        random_state: int = 207,
        use_smote: bool = False,
        impute_strategy: str = "mean",  # "mean" or "subgroup_mean"
        smote_strategy: str = "smote",
        smote_sampling_strategy: str = "auto",

        # Parameters for SMOTE pre-sampling
        smote_minority_n: int = 2000,
        smote_majority_cap: int = 8000,

        # Output format for different model types
        output_format: str = "onehot",  # "onehot" or "embedding"
) -> Dict[str, Any]:
    """
    End-to-end preprocessing pipeline built on top of Shanti's notebook, starting from raw CSV. This notebook has some
    memory logs to track RAM consumption during runtime due to prior OOM issues

    Parameters
    ----------
    n_per_class : int
        Number of samples per class when NOT using SMOTE (balanced sampling)
    random_state : int
        Random seed for reproducibility
    use_smote : bool
        Whether to apply SMOTE/SMOTENC for class balancing
    impute_strategy : str
        "mean" or "subgroup_mean"
    smote_strategy : str
        "smote" (standard, after one-hot) or "smotenc" (handles categoricals natively)
    smote_sampling_strategy : str or dict
        Passed to SMOTE. Use "auto" for full balancing, or dict for custom targets.
    smote_minority_n : int
        When use_smote=True, sample up to this many from minority classes before SMOTE
    smote_majority_cap : int
        When use_smote=True, cap majority class at this size before SMOTE
    output_format : str
        "onehot" - One-hot encode categoricals (for FFNN, tree models)
        "embedding" - Label encode categoricals (for embedding neural networks)

    Returns
    -------
    dict
        Contains X_train_std, X_val_std, X_test_std, Y_train, Y_val, Y_test, and metadata.

        For output_format="embedding", additionally contains:
        - X_train_cat, X_val_cat, X_test_cat (integer-encoded categoricals)
        - X_train_cont, X_val_cont, X_test_cont (standardized continuous)
        - categorical_info (dict with num_categories, embed_dim for each categorical)
        - label_encoders (dict of fitted LabelEncoders)
    """
    _log_memory("Pipeline start")

    # 1. Load raw data
    df_init = load_data_from_csv(convert_to_pandas=True)
    if df_init is None:
        raise RuntimeError("Failed to load data from CSV via data_loader.load_data_from_csv.")

    _log_memory("After data load")
    print(f"Loaded {len(df_init)} rows")

    # 2. Drop columns with >80% missing
    df = _drop_high_missing_columns(df_init)
    del df_init
    gc.collect()

    # 3. Create FIRE_SIZE_LABEL
    df = _create_fire_size_label(df)

    # 4. Sampling strategy - ALWAYS sample before SMOTE to avoid OOM
    if use_smote:
        # Pre-sample with stratified approach for SMOTE
        print(f"\nPre-sampling for SMOTE (minority_n={smote_minority_n}, majority_cap={smote_majority_cap})...")
        df_mini = _sample_stratified_for_smote(
            df,
            minority_n=smote_minority_n,
            majority_cap=smote_majority_cap,
            random_state=random_state
        )
    else:
        # Standard balanced sampling
        df_mini = _sample_balanced_classes(df, n_per_class=n_per_class, random_state=random_state)

    # Free the full dataset
    del df
    gc.collect()
    _log_memory("After sampling")

    # 5. Map FIRE_SIZE_LABEL to numeric
    df_mini = _encode_fire_size_label_numeric(df_mini)

    # 6. Shuffle rows
    rng = np.random.RandomState(random_state)
    shuffled_idx = rng.permutation(df_mini.index)
    df_mini = df_mini.loc[shuffled_idx].reset_index(drop=True)

    # 7. Train/val/test split (60/20/20)
    stratify_cols = ["FIRE_YEAR"]
    if use_smote:
        stratify_cols.append("FIRE_SIZE_LABEL")

    df_train, df_val, df_test = _train_val_test_split(
        df_mini,
        random_state=random_state,
        stratify_cols=stratify_cols
    )

    del df_mini
    gc.collect()

    # 8. Select features of interest
    tgt_feat_selected = _select_features(df_train)
    df_train = df_train[tgt_feat_selected]
    df_val = df_val[tgt_feat_selected]
    df_test = df_test[tgt_feat_selected]

    # Reduce floats to 32 to reduce memory
    for data in [df_train, df_val, df_test]:
        cols_float = data.select_dtypes(include=['float64']).columns
        data[cols_float] = data[cols_float].astype('float32')
    gc.collect()

    # 9. Impute selected vars with zero
    df_train = _impute_with_zero(df_train)
    df_val = _impute_with_zero(df_val)
    df_test = _impute_with_zero(df_test)

    # 10. Impute mean for GRIDMET + SDI + EPL_PCI + NDVI-1day
    feat_gridmet = [c for c in df_train.columns if "_5D_" in c]

    if impute_strategy == "subgroup_mean":
        print("Imputing with subgroup mean (GACC_PL)...")
        df_train, df_val, df_test = _impute_with_subgroup_mean(
            df_train, df_val, df_test, feat_gridmet, subgroup_col="GACC_PL"
        )
    else:
        print("Imputing with global mean...")
        df_train, df_val, df_test = _impute_with_mean(df_train, df_val, df_test, feat_gridmet)

    # 11. Clean NWCG_CAUSE_CLASSIFICATION
    df_train, df_val, df_test = _clean_nwcg_cause_classification([df_train, df_val, df_test])

    # 12. Apply SMOTE
    if use_smote and smote_strategy == "smotenc":
        print("\nApplying SMOTENC on pre-sampled data...")
        df_train = _apply_smotenc(
            df_train,
            random_state=random_state,
            sampling_strategy=smote_sampling_strategy
        )
    cat_cols = ["NWCG_CAUSE_CLASSIFICATION", "GACC_PL"]

    if output_format == "embedding":
        print("\nPreparing data for embedding model (label-encoded categoricals)...")

        df_train, df_val, df_test, label_encoders = _label_encode_categoricals(
            df_train, df_val, df_test, cat_cols=cat_cols
        )
        gc.collect()

        # Standardize and return embedding with appropriate format
        result = _standardize_features_for_embeddings(
            df_train, df_val, df_test, cat_cols, label_encoders
        )

        # Also provide combined arrays for compatibility
        if result["X_train_cat"] is not None:
            result["X_train_std"] = np.concatenate([
                result["X_train_cont"],
                result["X_train_cat"].astype('float32')
            ], axis=1)
            result["X_val_std"] = np.concatenate([
                result["X_val_cont"],
                result["X_val_cat"].astype('float32')
            ], axis=1)
            result["X_test_std"] = np.concatenate([
                result["X_test_cont"],
                result["X_test_cat"].astype('float32')
            ], axis=1)
        else:
            result["X_train_std"] = result["X_train_cont"]
            result["X_val_std"] = result["X_val_cont"]
            result["X_test_std"] = result["X_test_cont"]

    else:
        # 13. One-Hot Encode
        df_train, df_val, df_test = _one_hot_encode(df_train, df_val, df_test)
        gc.collect()

        # 14. Standard SMOTE (after one-hot encoding)
        if use_smote and smote_strategy == "smote":
            print("\nApplying standard SMOTE...")
            df_train = _apply_smote(
                df_train,
                random_state=random_state,
                sampling_strategy=smote_sampling_strategy
            )

        # 15. Standardize continuous features and return arrays/metadata
        result = _standardize_features(df_train, df_val, df_test)

    result["df_train_processed"] = df_train
    result["df_val_processed"] = df_val
    result["df_test_processed"] = df_test
    result["output_format"] = output_format

    _log_memory("Pipeline complete")

    return result


if __name__ == "__main__":
    # Test run with SMOTENC and EMBEDDING output
    print("Running preprocessing pipeline with SMOTENC + EMBEDDING format")

    data = build_preprocessed_data(
        use_smote=True,
        smote_strategy='smotenc',
        impute_strategy='subgroup_mean',
        smote_minority_n=50000,
        smote_majority_cap=100000,
        output_format='embedding',  # NEW: Get embedding-ready format
    )

    print("\n" + "=" * 60)
    print("Output Summary")
    print("=" * 60)

    print(f"\nCategorical features (integer-encoded):")
    if data["X_train_cat"] is not None:
        print(f"  Shape: {data['X_train_cat'].shape}")
        print(f"  Columns: {data['cols_categorical']}")
        print(f"  Sample values:\n{data['X_train_cat'][:3]}")

    print(f"\nContinuous features (standardized):")
    print(f"  Shape: {data['X_train_cont'].shape}")
    print(f"  Columns: {data['cols_continuous'][:5]}... ({len(data['cols_continuous'])} total)")

    print(f"\nCategorical info for embedding layers:")
    for col, info in data['categorical_info'].items():
        print(f"  {col}: {info['num_categories']} categories -> {info['embed_dim']}D embedding")

    print(f"\nLabels:")
    print(f"  Y_train: {data['Y_train'].shape}, distribution: {np.bincount(data['Y_train'])}")
    print(f"  Y_val: {data['Y_val'].shape}")
    print(f"  Y_test: {data['Y_test'].shape}")