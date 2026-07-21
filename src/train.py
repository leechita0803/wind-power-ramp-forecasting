from sklearn.svm import SVR
from xgboost import XGBRegressor
import joblib
from src.config import MODELS_DIR

def train_models(X_train, imf_df):
    """
    Train one model for each IMF.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Create models
    model1 = SVR()
    model2 = SVR()
    model3 = XGBRegressor()

    # Train models
    model1.fit(X_train, imf_df["imf1"])
    model2.fit(X_train, imf_df["imf2"])
    model3.fit(X_train, imf_df["imf3"])

    # Save models
    joblib.dump(model1, MODELS_DIR / "imf1_model.pkl")
    joblib.dump(model2, MODELS_DIR / "imf2_model.pkl")
    joblib.dump(model3, MODELS_DIR / "imf3_model.pkl")

    return {
        "imf1": model1,
        "imf2": model2,
        "imf3": model3
    }

def predict_models(models, X_test):
    """
    Generate predictions for each IMF model.
    """
    pred1 = models["imf1"].predict(X_test)
    pred2 = models["imf2"].predict(X_test)
    pred3 = models["imf3"].predict(X_test)

    return {
        "imf1": pred1,
        "imf2": pred2,
        "imf3": pred3
    }