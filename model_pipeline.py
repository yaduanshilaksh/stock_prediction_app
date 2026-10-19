import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score
from typing import Dict, Tuple, Any

FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume', 
    'Daily_Return', 'Volatility_20', 'SMA_10', 
    'SMA_50', 'MACD', 'MACD_Signal', 'RSI_14'
]

def train_and_evaluate_regressors(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Trains Linear Regression, Random Forest, and XGBoost Regressors on historical price data.
    """
    X = df[FEATURE_COLS]
    y = df['Target_Price']

    # Sequential Train-Test Split (Preserves time order)
    split_idx = int(len(df) * train_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    }

    results = {}
    preds_df = pd.DataFrame(index=y_test.index)
    preds_df['Actual'] = y_test

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds_df[f'{name}_Pred'] = preds
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        results[name] = {"RMSE": rmse, "R2": r2, "model_obj": model}

    return results, preds_df


def train_and_evaluate_classifiers(df: pd.DataFrame, train_ratio: float = 0.8) -> Dict[str, Dict[str, float]]:
    """
    Trains Logistic Regression, Random Forest, and XGBoost Classifiers for directional movement prediction.
    """
    X = df[FEATURE_COLS]
    y = df['Target_Dir']

    split_idx = int(len(df) * train_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42, eval_metric='logloss')
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        results[name] = {"Accuracy": acc, "Precision": prec}

    return results