"""STEP 4 - questionnaire model (your existing ML project, leakage removed).
Run:  python src/train_questionnaire.py
"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import config

if not config.QUEST_CSV.exists():
    raise SystemExit(f"Missing {config.QUEST_CSV}\nCopy autism_screening.csv there.")
d = pd.read_csv(config.QUEST_CSV).drop_duplicates()
# NOTE: 'result' (sum of A1-A10) is dropped on purpose. In this dataset Class/ASD == YES exactly when
# result >= 7, so keeping it is label leakage (fake ~100% accuracy).
d["age"] = pd.to_numeric(d["age"], errors="coerce")
d["age"] = d["age"].fillna(d["age"].median())
d["gender"] = (d["gender"].astype(str).str.lower() == "m").astype(int)
d["jundice"] = (d["jundice"].astype(str).str.lower() == "yes").astype(int)
d["austim"] = (d["austim"].astype(str).str.lower() == "yes").astype(int)
features = [f"A{i}_Score" for i in range(1, 11)] + ["age", "gender", "jundice", "austim"]
X, y = d[features], (d["Class/ASD"].astype(str).str.upper() == "YES").astype(int)

model = Pipeline([("scale", StandardScaler()),
                  ("vote", VotingClassifier([
                      ("lr", LogisticRegression(max_iter=1000, random_state=config.SEED)),
                      ("rf", RandomForestClassifier(random_state=config.SEED)),
                      ("svm", SVC(probability=True, random_state=config.SEED))], voting="soft"))])
print("5-fold CV accuracy:", round(cross_val_score(model, X, y, cv=5).mean(), 4))
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=config.SEED)
model.fit(Xtr, ytr)
p = model.predict_proba(Xte)[:, 1]
pred = (p >= 0.5).astype(int)
print("hold-out:", dict(Accuracy=round(accuracy_score(yte, pred), 3), Precision=round(precision_score(yte, pred), 3),
                        Recall=round(recall_score(yte, pred), 3), F1=round(f1_score(yte, pred), 3),
                        ROC_AUC=round(roc_auc_score(yte, p), 3)))
print("NOTE: A1-A10 still define the label in this dataset, so high accuracy is expected. Say so in the report.")
model.fit(X, y)                                   # final model on all rows
joblib.dump({"model": model, "features": features}, config.MODEL_DIR / "questionnaire_model.joblib")
print("Saved:", config.MODEL_DIR / "questionnaire_model.joblib")
print("Next:  streamlit run app.py")
