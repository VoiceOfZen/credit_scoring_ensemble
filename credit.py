import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from data_utils import load_and_prepare_data
from models import get_models

def save_roc_curve(fpr, tpr, model_name):
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve ({model_name})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    filename = f'roc_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename)
    plt.close()
    print(f"[✔] Saved ROC curve: {filename}")

def save_feature_importance(model, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = importances.argsort()[::-1]

        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importances - {model_name}')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()

        filename = f'feat_imp_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename)
        plt.close()
        print(f"[✔] Saved feature importance plot: {filename}")

def evaluate_model(name, model, X_test, y_test, feature_names):
    print(f"\n=== {name} ===")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC: {roc_auc:.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    save_roc_curve(fpr, tpr, name)
    save_feature_importance(model, feature_names, name)

def main():
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()
    models = get_models()

    for name, model in models.items():
        model.fit(X_train, y_train)
        evaluate_model(name, model, X_test, y_test, feature_names)

if __name__ == "__main__":
    main()
