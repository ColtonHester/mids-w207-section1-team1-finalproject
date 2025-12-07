import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC


def load_and_prepare_data_dropmissing(csv_path, smote_strategy="smotenc", sample_size=500_000):

    print("\nLoading with Polars...")
    df = pl.read_csv(csv_path, infer_schema_length=20000, ignore_errors=True)

    # Parse date
    if "DISCOVERY_DATE" in df.columns:
        df = df.with_columns(
            pl.col("DISCOVERY_DATE").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("DISCOVERY_DATE")
        )

    df = df.to_pandas()
    print(f"➡ Loaded: {df.shape}")


    # Feature Engineering

    # FIRE_SIZE_LABEL
    if "FIRE_SIZE" in df.columns:
        bins = [0, 100, 4999, 29000, df["FIRE_SIZE"].max()]
        labels = ["small", "medium", "large", "very large"]
        mapping = {"small": 0, "medium": 1, "large": 2, "very large": 3}
        df["FIRE_SIZE_LABEL"] = pd.cut(df["FIRE_SIZE"], bins=bins, labels=labels).map(mapping)
        df = df.drop(columns=["FIRE_SIZE"], errors="ignore")
    else:
        raise KeyError("FIRE_SIZE column missing — cannot create FIRE_SIZE_LABEL")

    # IS_WEST_OF_100
    if "LONGITUDE" in df.columns:
        df["IS_WEST_OF_100"] = np.where(df["LONGITUDE"] < -100, 1, 0)
        df = df.drop(columns=["LATITUDE", "LONGITUDE"], errors="ignore")
    else:
        df["IS_WEST_OF_100"] = 0  # fallback

    # Define selected features (AFTER creation)
    trgt_feat_fod = ["FIRE_SIZE_LABEL", "FIRE_YEAR", "DISCOVERY_DOY",
                     "NWCG_CAUSE_CLASSIFICATION", "IS_WEST_OF_100"]

    feat_gridmet = [col for col in df.columns if "_5D_" in col]
    feat_rmgmta = ["SDI"] if "SDI" in df.columns else []
    feat_firestation = ["No_FireStation_20.0km"] if "No_FireStation_20.0km" in df.columns else []
    feat_gacc = ["GACC_PL"] if "GACC_PL" in df.columns else []
    feat_ghm = ["GHM"] if "GHM" in df.columns else []
    feat_ndvi = ["NDVI-1day"] if "NDVI-1day" in df.columns else []
    feat_npl = ["NPL"] if "NPL" in df.columns else []
    feat_svi = ["EPL_PCI"] if "EPL_PCI" in df.columns else []
    feat_rangeland = [col for col in ["rpms", "rpms_1km"] if col in df.columns]

    trgt_feat_selected = list(set(
        trgt_feat_fod + feat_gridmet + feat_rmgmta + feat_firestation +
        feat_gacc + feat_ghm + feat_ndvi + feat_npl + feat_svi + feat_rangeland
    ))

    df = df[trgt_feat_selected]

    # Drop Missing Rows
    print("\n🧹 Dropping rows with ANY missing value...")
    before = df.shape[0]
    df = df.dropna(axis=0, how="any")
    after = df.shape[0]
    print(f"➡ Removed {before - after:,} rows — Remaining: {after:,}")

    # Train/Val/Test Split
    val_size = 0.2 / (0.6 + 0.2)

    df_train_main, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=207,
        stratify=df["FIRE_SIZE_LABEL"]
    )

    df_train, df_val = train_test_split(
        df_train_main,
        test_size=val_size,
        random_state=207,
        stratify=df_train_main["FIRE_SIZE_LABEL"]
    )

    # Optional Downsampling
    if sample_size and df_train.shape[0] > sample_size:
        df_train = df_train.sample(n=sample_size, random_state=207)
        print(f"✂ Training down-sampled ➜ {df_train.shape}")

    # Split Features / Targets
    Y_train = df_train["FIRE_SIZE_LABEL"]
    Y_val = df_val["FIRE_SIZE_LABEL"]
    Y_test = df_test["FIRE_SIZE_LABEL"]

    X_train = df_train.drop(columns=["FIRE_SIZE_LABEL"])
    X_val = df_val.drop(columns=["FIRE_SIZE_LABEL"])
    X_test = df_test.drop(columns=["FIRE_SIZE_LABEL"])

    # SMOTENC
    cat_cols = [c for c in ["NWCG_CAUSE_CLASSIFICATION", "GACC_PL", "IS_WEST_OF_100"] if c in X_train.columns]
    cat_indices = [X_train.columns.get_loc(col) for col in cat_cols]

    if smote_strategy.lower() == "smotenc":
        print("\nApplying SMOTENC...")
        sm = SMOTENC(categorical_features=cat_indices, random_state=207)
        X_train_resampled, Y_train_resampled = sm.fit_resample(X_train, Y_train)
    else:
        X_train_resampled, Y_train_resampled = X_train, Y_train

    # One-hot encode categorical features
    X_train_ohe = pd.get_dummies(X_train_resampled, columns=cat_cols, drop_first=True)
    X_val_ohe = pd.get_dummies(X_val, columns=cat_cols, drop_first=True)
    X_test_ohe = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

    train_cols = X_train_ohe.columns
    X_val_ohe = X_val_ohe.reindex(columns=train_cols, fill_value=0)
    X_test_ohe = X_test_ohe.reindex(columns=train_cols, fill_value=0)

    return (
        X_train_ohe.values.astype("float32"),
        Y_train_resampled.astype("int32"),
        X_val_ohe.values.astype("float32"),
        Y_val.astype("int32"),
        X_test_ohe.values.astype("float32"),
        Y_test.astype("int32")
    )
