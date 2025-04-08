import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define paths
BASE_DIR = '/home/ubuntu/grammar_scoring'
FEATURES_DIR = os.path.join(BASE_DIR, 'features')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Load features and labels
train_features = np.load(os.path.join(FEATURES_DIR, 'train_features.npy'))
train_labels = np.load(os.path.join(FEATURES_DIR, 'train_labels.npy'))
val_features = np.load(os.path.join(FEATURES_DIR, 'val_features.npy'))
val_labels = np.load(os.path.join(FEATURES_DIR, 'val_labels.npy'))
test_features = np.load(os.path.join(FEATURES_DIR, 'test_features.npy'))

# Load feature column names and test filenames
with open(os.path.join(FEATURES_DIR, 'feature_columns.pkl'), 'rb') as f:
    feature_columns = pickle.load(f)

with open(os.path.join(FEATURES_DIR, 'test_filenames.pkl'), 'rb') as f:
    test_filenames = pickle.load(f)

print(f"Training features shape: {train_features.shape}")
print(f"Validation features shape: {val_features.shape}")
print(f"Test features shape: {test_features.shape}")
print(f"Number of feature columns: {len(feature_columns)}")

# Define evaluation metrics
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    return {
        'model_name': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

# Function to plot predictions vs actual
def plot_predictions(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('Actual Grammar Score')
    plt.ylabel('Predicted Grammar Score')
    plt.title(f'{model_name} - Predictions vs Actual')
    plt.grid(True)
    plt.savefig(os.path.join(MODELS_DIR, f'{model_name}_predictions.png'))
    plt.close()

# 1. Train baseline models
print("\n=== Training Baseline Models ===\n")

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(train_features, train_labels)
ridge_val_pred = ridge.predict(val_features)
ridge_metrics = evaluate_model(val_labels, ridge_val_pred, "Ridge Regression")
plot_predictions(val_labels, ridge_val_pred, "Ridge Regression")

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(train_features, train_labels)
rf_val_pred = rf.predict(val_features)
rf_metrics = evaluate_model(val_labels, rf_val_pred, "Random Forest")
plot_predictions(val_labels, rf_val_pred, "Random Forest")

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(train_features, train_labels)
gb_val_pred = gb.predict(val_features)
gb_metrics = evaluate_model(val_labels, gb_val_pred, "Gradient Boosting")
plot_predictions(val_labels, gb_val_pred, "Gradient Boosting")

# SVR
svr = SVR(kernel='rbf')
svr.fit(train_features, train_labels)
svr_val_pred = svr.predict(val_features)
svr_metrics = evaluate_model(val_labels, svr_val_pred, "SVR")
plot_predictions(val_labels, svr_val_pred, "SVR")

# MLP Regressor
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
mlp.fit(train_features, train_labels)
mlp_val_pred = mlp.predict(val_features)
mlp_metrics = evaluate_model(val_labels, mlp_val_pred, "MLP Regressor")
plot_predictions(val_labels, mlp_val_pred, "MLP Regressor")

# 2. Neural Network with TensorFlow/Keras
print("\n=== Training Neural Network ===\n")

# Define the model
def create_nn_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Train the neural network
nn_model = create_nn_model(train_features.shape[1])

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODELS_DIR, 'nn_model_best.h5'),
    monitor='val_loss',
    save_best_only=True
)

# Train the model
history = nn_model.fit(
    train_features, train_labels,
    validation_data=(val_features, val_labels),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

# Evaluate the neural network
nn_val_pred = nn_model.predict(val_features).flatten()
nn_metrics = evaluate_model(val_labels, nn_val_pred, "Neural Network")
plot_predictions(val_labels, nn_val_pred, "Neural Network")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'nn_training_history.png'))
plt.close()

# 3. Compare all models
print("\n=== Model Comparison ===\n")

models_metrics = [ridge_metrics, rf_metrics, gb_metrics, svr_metrics, mlp_metrics, nn_metrics]
models_df = pd.DataFrame(models_metrics)
models_df = models_df.sort_values('rmse')
print(models_df)

# Save the comparison results
models_df.to_csv(os.path.join(MODELS_DIR, 'model_comparison.csv'), index=False)

# Plot model comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(models_df['model_name'], models_df['rmse'])
plt.title('RMSE by Model')
plt.xticks(rotation=45)
plt.ylabel('RMSE (lower is better)')

plt.subplot(1, 2, 2)
plt.bar(models_df['model_name'], models_df['r2'])
plt.title('R² by Model')
plt.xticks(rotation=45)
plt.ylabel('R² (higher is better)')

plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'model_comparison.png'))
plt.close()

# 4. Save the best model
best_model_name = models_df.iloc[0]['model_name']
print(f"\nBest model: {best_model_name} with RMSE: {models_df.iloc[0]['rmse']:.4f}")

# Determine which model is best and save it
if best_model_name == "Ridge Regression":
    best_model = ridge
    with open(os.path.join(MODELS_DIR, 'best_model.pkl'), 'wb') as f:
        pickle.dump(ridge, f)
elif best_model_name == "Random Forest":
    best_model = rf
    with open(os.path.join(MODELS_DIR, 'best_model.pkl'), 'wb') as f:
        pickle.dump(rf, f)
elif best_model_name == "Gradient Boosting":
    best_model = gb
    with open(os.path.join(MODELS_DIR, 'best_model.pkl'), 'wb') as f:
        pickle.dump(gb, f)
elif best_model_name == "SVR":
    best_model = svr
    with open(os.path.join(MODELS_DIR, 'best_model.pkl'), 'wb') as f:
        pickle.dump(svr, f)
elif best_model_name == "MLP Regressor":
    best_model = mlp
    with open(os.path.join(MODELS_DIR, 'best_model.pkl'), 'wb') as f:
        pickle.dump(mlp, f)
elif best_model_name == "Neural Network":
    best_model = nn_model
    nn_model.save(os.path.join(MODELS_DIR, 'best_model_nn.h5'))
    best_model_is_nn = True

# 5. Generate predictions on test set
print("\n=== Generating Test Predictions ===\n")

# Use the best model to predict on test set
if best_model_name == "Neural Network":
    test_predictions = nn_model.predict(test_features).flatten()
else:
    test_predictions = best_model.predict(test_features)

# Create submission file
submission_df = pd.DataFrame({
    'filename': test_filenames,
    'label': test_predictions
})

# Ensure predictions are within the valid range (1.0 to 5.0)
submission_df['label'] = submission_df['label'].clip(1.0, 5.0)

# Save submission file
submission_df.to_csv(os.path.join(MODELS_DIR, 'submission.csv'), index=False)
print(f"Submission file saved with {len(submission_df)} predictions")

print("\nModel training and evaluation completed successfully!")
