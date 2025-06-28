import joblib
import os
from sklearn.metrics import roc_auc_score

def evaluate_and_compare(X_test, y_test):
    try:
        latest_model = joblib.load('models/latest_model.pkl')
        prev_model = joblib.load('models/previous_model.pkl')
    except FileNotFoundError:
        print("ძველი მოდელი არ არსებობს")
        return True

    # AUC შეფასება
    latest_score = roc_auc_score(y_test, latest_model.predict_proba(X_test)[:, 1])
    prev_score = roc_auc_score(y_test, prev_model.predict_proba(X_test)[:, 1])

    if latest_score > prev_score:
        print(f"✅ ახალი მოდელი უკეთესია: {latest_score:.4f} > {prev_score:.4f}")
        os.rename('models/latest_model.pkl', 'models/previous_model.pkl')
        return True
    else:
        print(f"❌ ძველი მოდელი უკეთესია: {prev_score:.4f} ≥ {latest_score:.4f}")
        os.remove('models/latest_model.pkl')
        return False