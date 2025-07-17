import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
CSV_FILE = 'asl_landmarks_800.csv'
data = pd.read_csv(CSV_FILE)

# Separate features and labels
X = data.iloc[:, :-1]  # All columns except the last one (features)
y = data.iloc[:, -1]   # Last column (labels)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define XGBoost model
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=26, eval_metric='mlogloss')

# Hyperparameter tuning
param_grid = {
    'max_depth': [4, 6, 5],
    'learning_rate': [0.05, 0.1, 0.2, 0.3],
    'n_estimators': [50, 100, 150, 200, 250],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'gamma': [0, 0.1, 0.3, 0.5, 1]  # Adding gamma regularization
}
scoring_metrics = ['accuracy', 'neg_log_loss', 'f1_weighted']

grid_search = GridSearchCV(
    xgb_model, param_grid, scoring=scoring_metrics, refit='accuracy', cv=3, verbose=10, n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Train best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the trained model
best_model.save_model('xgboost_asl_model_scoring.json')

print("Best parameters:", grid_search.best_params_)
print("Trained XGBoost model saved successfully!")
