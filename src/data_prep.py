import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTENC, RandomOverSampler


# ------------------------------------------
# SMOTE Strategy
# ------------------------------------------
def apply_smote_strategy(X_train, Y_train, strategy, cat_indices):

    strategy = strategy.lower() if isinstance(strategy, str) else "none"

    if strategy == "smotenc":
        print("\nApplying SMOTENC oversampling")
        sm = SMOTENC(categorical_features=cat_indices, random_state=207)
        return sm.fit_resample(X_train, Y_train)

    elif strategy == "random":
        print("\nApplying RandomOverSampler")
        ro = RandomOverSampler(random_state=207)
        return ro.fit_resample(X_train, Y_train)

    else:
        print("\n No oversampling applied (dataset remains imbalanced).")
        return X_train.copy(), Y_train.copy()


# ------------------------------------------
# MAIN DATA PREP FUNCTION
# ------------------------------------------
def load_and_prepare_data(csv_path, smote_strategy="smotenc"):

    print("\nLoading data with Polars...")
    df = pl.read_csv(csv_path, infer_schema_length=10000, ignore_errors=True)

    df = df.with_columns(
        pl.col("DISCOVERY_DATE").str.strptime(pl.Date, "%Y-%m-%d").alias("DISCOVERY_DATE")
    )

    df = df.to_pandas()
    print("File loaded into Pandas. Shape:", df.shape)

    # Drop >80% missing columns
    cutoff_missing = df.shape[0] * 0.8
    cols_drop = df.columns[df.isna().sum() > cutoff_missing]
    df = df.drop(columns=cols_drop)

    # FIRE_SIZE binning
    bins = [0, 100, 4999, 29000, df["FIRE_SIZE"].max()]
    labels = ["small", "medium", "large", "very large"]
    mapping = {"small": 0, "medium": 1, "large": 2, "very large": 3}
    df["FIRE_SIZE_LABEL"] = pd.cut(df["FIRE_SIZE"], bins=bins, labels=labels).map(mapping)

    # longitude-based feature
    df["IS_WEST_OF_100"] = np.where(df["LONGITUDE"] < -100, 1, 0)
    df = df.drop(columns=["LATITUDE", "LONGITUDE"], errors="ignore")

    # ---- Train / Val / Test ----
    np.random.seed(207)
    val_size = 0.2 / (0.2 + 0.6)

    df_train_main, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=207,
        stratify=list(zip(df["FIRE_YEAR"], df["FIRE_SIZE_LABEL"]))
    )

    df_train, df_val = train_test_split(
        df_train_main,
        test_size=val_size,
        random_state=207,
        stratify=list(zip(df_train_main["FIRE_YEAR"], df_train_main["FIRE_SIZE_LABEL"]))
    )

    # ---- Feature selection ----
    trgt_feat_fod = ["FIRE_SIZE_LABEL", "FIRE_YEAR", "DISCOVERY_DOY", "NWCG_CAUSE_CLASSIFICATION", "IS_WEST_OF_100"]
    feat_gridmet = [col for col in df_train.columns if "_5D_" in col]
    feat_rmgmta = ["SDI"]
    feat_firestation = ["No_FireStation_20.0km"]
    feat_gacc = ["GACC_PL"]
    feat_ghm = ["GHM"]
    feat_ndvi = ["NDVI-1day"]
    feat_npl = ["NPL"]
    feat_svi = ["EPL_PCI"]
    feat_rangeland = ["rpms", "rpms_1km"]

    trgt_feat_selected = list(set(
        trgt_feat_fod + feat_gridmet + feat_rmgmta + feat_firestation +
        feat_gacc + feat_ghm + feat_ndvi + feat_npl + feat_svi + feat_rangeland
    ))

    df_train = df_train[trgt_feat_selected]
    df_val = df_val[trgt_feat_selected]
    df_test = df_test[trgt_feat_selected]

    # ---- Imputation ----
    for data in [df_train, df_val, df_test]:
        data["No_FireStation_20.0km"] = data["No_FireStation_20.0km"].fillna(0)
        data["GACC_PL"] = data["GACC_PL"].fillna(0)
        data["EPL_PCI"] = data["EPL_PCI"].replace(-999, np.nan)

    cols_impute_mean = feat_gridmet + ["SDI", "EPL_PCI", "NDVI-1day"]
    imp = SimpleImputer(strategy="mean")
    df_train[cols_impute_mean] = imp.fit_transform(df_train[cols_impute_mean])
    df_val[cols_impute_mean] = imp.transform(df_val[cols_impute_mean])
    df_test[cols_impute_mean] = imp.transform(df_test[cols_impute_mean])

    # Simplify cause category
    mapping_cause = {"Human": "hum", "Natural": "nat", "Missing data/not specified/undetermined": "miss-unspec"}
    for data in [df_train, df_val, df_test]:
        data["NWCG_CAUSE_CLASSIFICATION"] = data["NWCG_CAUSE_CLASSIFICATION"].map(mapping_cause)

    # Outcome/features
    Y_train = df_train["FIRE_SIZE_LABEL"]
    Y_val = df_val["FIRE_SIZE_LABEL"]
    Y_test = df_test["FIRE_SIZE_LABEL"]

    X_train = df_train.drop(columns=["FIRE_SIZE_LABEL"])
    X_val = df_val.drop(columns=["FIRE_SIZE_LABEL"])
    X_test = df_test.drop(columns=["FIRE_SIZE_LABEL"])

    # ---- PRE-SAMPLING BEFORE SMOTENC ----
    if smote_strategy.lower() == "smotenc":
        print("\nPre-Sampling before SMOTENC (500k small class)...")

        df_temp = X_train.copy()
        df_temp["target"] = Y_train

        PRE_SAMPLE_SMALL = 500_000

        counts = df_temp["target"].value_counts()
        print("\nOriginal class distribution:")
        print(counts)

        dfs = []
        for cls, count in counts.items():
            if cls == 0:  # small fires
                sampled = df_temp[df_temp["target"] == cls].sample(
                    PRE_SAMPLE_SMALL, random_state=207
                )
            else:
                sampled = df_temp[df_temp["target"] == cls]
            dfs.append(sampled)

        df_presampled = pd.concat(dfs).reset_index(drop=True)
        print(f"\nPre-sampled total rows: {len(df_presampled):,}")

        X_train = df_presampled.drop(columns=["target"])
        Y_train = df_presampled["target"]

    # ---- Apply SMOTE/SMOTENC/Random ----
    cat_cols = ["NWCG_CAUSE_CLASSIFICATION", "GACC_PL", "IS_WEST_OF_100"]
    cat_indices = [X_train.columns.get_loc(col) for col in cat_cols]

    X_train_resampled, Y_train_resampled = apply_smote_strategy(
        X_train, Y_train, smote_strategy, cat_indices
    )

    # ---- One-hot encode ----
    X_train_ohe = pd.get_dummies(X_train_resampled, columns=cat_cols, drop_first=True)
    X_val_ohe = pd.get_dummies(X_val, columns=cat_cols, drop_first=True)
    X_test_ohe = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

    # Align columns
    train_cols = X_train_ohe.columns
    X_val_ohe = X_val_ohe.reindex(columns=train_cols, fill_value=0)
    X_test_ohe = X_test_ohe.reindex(columns=train_cols, fill_value=0)

    # ---- Standardize ----
    scaler = StandardScaler()
    continuous = [c for c in X_train.columns if c not in cat_cols]

    X_train_std = scaler.fit_transform(X_train_ohe[continuous])
    X_val_std = scaler.transform(X_val_ohe[continuous])
    X_test_std = scaler.transform(X_test_ohe[continuous])

    X_train_final = np.concatenate(
        [X_train_std, X_train_ohe.drop(columns=continuous)], axis=1
    ).astype("float32")

    X_val_final = np.concatenate(
        [X_val_std, X_val_ohe.drop(columns=continuous)], axis=1
    ).astype("float32")

    X_test_final = np.concatenate(
        [X_test_std, X_test_ohe.drop(columns=continuous)], axis=1
    ).astype("float32")

    return (
        X_train_final,
        Y_train_resampled.astype("int32"),
        X_val_final,
        Y_val.astype("int32"),
        X_test_final,
        Y_test.astype("int32")
    )