# main_model_run.py

from src.modeling import (
    train_random_forest, evaluate_random_forest,
    build_fnn, train_fnn, evaluate_fnn
)

# X_train_final, Y_train_resampled, X_val_final, Y_val

print("Training RF...")
rf_model = train_random_forest(X_train_final, Y_train_resampled)
evaluate_random_forest(rf_model, X_val_final, Y_val)

print("\nTraining FNN...")
fnn = build_fnn(input_shape=X_train_final.shape[1])
fnn, _ = train_fnn(fnn, X_train_final, Y_train_resampled, X_val_final, Y_val)
evaluate_fnn(fnn, X_val_final, Y_val)
