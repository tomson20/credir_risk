import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from collections import Counter

# MLflow Tracking URI
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Credit Risk Model")

# მონაცემების ჩატვირთვა
df = pd.read_csv('data/train_data.csv')

# ნიშნებები და მიზანი
X = df.drop(['defaulted'], axis=1)
y = df['defaulted']

# შემოწმება: არის თუ არა საკმარისი კლასები stratify-ისთვის
class_counts = Counter(y)
if len(class_counts) < 2 or min(class_counts.values()) < 2:
    print(f"⚠️ Stratify ვერ მუშაობს → {class_counts}")
    stratify = None
else:
    stratify = y

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=stratify, random_state=42)

# მოდელის ტრენინგი
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# პროგნოზი
preds = model.predict_proba(X_test)[:, 1]

# მეტრიკების ლოგირება
with mlflow.start_run():
    mlflow.log_params(model.get_params())

    # შემოწმება: არის თუ არა ორივე კლასი y_test-ში
    if len(y_test.unique()) < 2:
        acc = accuracy_score(y_test, (preds > 0.5).astype(int))
        mlflow.log_metric("accuracy", acc)
        print(f"⚠️ ROC AUC ვერ გამოითვალა. გამოიყენებულია Accuracy: {acc:.4f}")
    else:
        auc = roc_auc_score(y_test, preds)
        mlflow.log_metric("roc_auc", auc)
        print(f"✅ ROC AUC: {auc:.4f}")

# მოდელის შენახვა
joblib.dump(model, 'models/best_model.pkl')
print("✅ მოდელი წარმატებით შეინახა.")