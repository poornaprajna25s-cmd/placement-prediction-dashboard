import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

def train_models():
    df = pd.read_csv("data.csv")

    # 🔹 Data Cleaning
    df.fillna(df.mean(), inplace=True)
    df.drop_duplicates(inplace=True)

    # 🔹 Features
    X = df.drop("Placed", axis=1)
    y = df["Placed"]

    # 🔹 Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 🔹 Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = {
        "Logistic": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    results = {}
    model_outputs = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        results[name] = acc

        cm = confusion_matrix(y_test, preds)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr, roc_auc = None, None, None

        model_outputs[name] = {
            "model": model,
            "cm": cm,
            "fpr": fpr,
            "tpr": tpr,
            "auc": roc_auc
        }

    best_model = max(results, key=results.get)

    return results, model_outputs, best_model, scaler