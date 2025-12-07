# src/modeling.py

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------
# RANDOM FOREST
# ---------------------------

def train_random_forest(X_train, y_train, n_estimators=50, random_state=42):
    """
    Train Random Forest classifier
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_random_forest(model, X_val, y_val):
    """
    Evaluate Random Forest on validation set
    """
    y_pred_prob = model.predict_proba(X_val)
    y_pred = np.argmax(y_pred_prob, axis=1)

    print("\nRandom Forest Validation Accuracy:", accuracy_score(y_val, y_pred))
    print("\nClassification Report:\n", classification_report(y_val, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))

    return y_pred


# ---------------------------
# FEEDFORWARD NEURAL NETWORK
# ---------------------------

def build_fnn(input_dim, hidden_layers=[64, 32], dropout_rate=0.2, learning_rate=0.01):
    """
    Build a Feedforward Neural Network model (FNN).

    Parameters:
        input_dim (int): Number of input features (columns).
        hidden_layers (list): List of neuron counts for each hidden layer.
        dropout_rate (float): Dropout applied after each hidden layer.
        learning_rate (float): Learning rate for Adam optimizer.

    Returns:
        model: Compiled Keras FNN model.
    """

    model = Sequential()

    # First hidden layer (requires input_dim)
    model.add(Dense(hidden_layers[0], activation="relu", input_shape=(input_dim,)))
    model.add(Dropout(dropout_rate))

    # Additional hidden layers
    for layer_size in hidden_layers[1:]:
        model.add(Dense(layer_size, activation="relu"))
        model.add(Dropout(dropout_rate))

    # Output layer — 4 classes
    model.add(Dense(4, activation="softmax"))

    # Compile Model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model



def train_fnn(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=64):
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    return model, history



def evaluate_fnn(model, X_val, y_val):
    """
    Evaluate Neural Network.
    Returns predictions so metrics can be stored.
    """
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import numpy as np

    y_pred_prob = model.predict(X_val)
    y_pred = np.argmax(y_pred_prob, axis=1)

    print("\nValidation Accuracy:", accuracy_score(y_val, y_pred))
    print("\nClassification Report:\n", classification_report(y_val, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))

    return y_pred
