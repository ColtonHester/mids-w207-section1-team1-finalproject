import numpy as np
import pandas as pd
from typing import Tuple, Union, Optional


def train_val_test_split(
        data: Union[np.ndarray, pd.DataFrame],
        train_pct: float = 0.6,
        val_pct: float = 0.2,
        test_pct: float = 0.2,
        shuffle: bool = False,
        random_state: Optional[int] = None,
        stratify: Optional[Union[np.ndarray, pd.Series]] = None
) -> Tuple:
    """
    Split data into train, validation, and test sets.

    Parameters:
    -----------
    data : np.ndarray or pd.DataFrame
        Input data to split
    train_pct : float
        Percentage for training set (default: 0.6)
    val_pct : float
        Percentage for validation set (default: 0.2)
    test_pct : float
        Percentage for test set (default: 0.2)
    shuffle : bool
        Whether to shuffle data before splitting (default: False)
    random_state : int, optional
        Random seed for reproducibility
    stratify : array-like, optional
        If not None, split in a stratified fashion using this as class labels

    Returns:
    --------
    train, val, test : same type as input data
    """
    # Validate percentages
    assert np.isclose(train_pct + val_pct + test_pct, 1.0), \
        f"Percentages must sum to 1.0, got {train_pct + val_pct + test_pct}"

    n = len(data)
    indices = np.arange(n)

    # Handle shuffling
    if shuffle:
        rng = np.random.RandomState(random_state)
        if stratify is not None:
            # Stratified shuffle
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, test_size=(val_pct + test_pct),
                                         random_state=random_state)
            train_idx, temp_idx = next(sss.split(indices, stratify))

            # Split temp into val and test
            val_size = val_pct / (val_pct + test_pct)
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - val_size),
                                          random_state=random_state)
            val_idx, test_idx = next(sss2.split(temp_idx, stratify[temp_idx]))
            val_idx = temp_idx[val_idx]
            test_idx = temp_idx[test_idx]
        else:
            rng.shuffle(indices)
            train_idx, val_idx, test_idx = _split_indices(indices, train_pct, val_pct)
    else:
        train_idx, val_idx, test_idx = _split_indices(indices, train_pct, val_pct)

    # Return splits based on data type
    if isinstance(data, pd.DataFrame):
        return data.iloc[train_idx], data.iloc[val_idx], data.iloc[test_idx]
    else:
        return data[train_idx], data[val_idx], data[test_idx]


def _split_indices(indices: np.ndarray, train_pct: float, val_pct: float) -> Tuple:
    """Helper function to split indices."""
    n = len(indices)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


if __name__ == "__main__":
    # With numpy array
    X = np.random.randn(1000, 10)
    X_train, X_val, X_test = train_val_test_split(X, 0.6, 0.2, 0.2, shuffle=True, random_state=42)

    # With pandas DataFrame
    df = pd.DataFrame(np.random.randn(1000, 5), columns=['A', 'B', 'C', 'D', 'E'])
    df_train, df_val, df_test = train_val_test_split(df, 0.7, 0.15, 0.15)

    # With stratification
    y = np.random.choice([0, 1, 2], size=1000)
    X_train, X_val, X_test = train_val_test_split(
        X, 0.6, 0.2, 0.2, shuffle=True, random_state=42, stratify=y
    )
    print(X_train[:5], X_val[:5], X_test[:5])