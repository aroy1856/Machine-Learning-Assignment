import xgboost as xgb
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

MODEL_PATH = "model/xgb_model.pkl"

def train_xgboost(X_train, y_train, scoring='recall'):
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    }
    model = xgb.XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    grid_search = GridSearchCV(
        model,
        param_grid,
        scoring=scoring,
        cv=5,
        verbose=2,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Save with pickle
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(grid_search.best_estimator_, f)

    return grid_search.best_estimator_

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred, average='weighted'),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_pred)
    }
