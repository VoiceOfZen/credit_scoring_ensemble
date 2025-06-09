import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay


def evaluate_models(models, X_train, X_test, y_train, y_test):
    """
    Обучает и оценивает переданные модели, строит ROC-кривые и выводит метрики.
    """
    results = {}
    plt.figure(figsize=(10, 8))

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred))
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC: {auc:.4f}")

        results[name] = auc
        RocCurveDisplay.from_predictions(y_test, y_proba, name=name, ax=plt.gca())

    plt.title("ROC-кривые моделей")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nСравнение ROC AUC:")
    for name, auc in results.items():
        print(f"{name}: {auc:.4f}")


def plot_feature_importances(model, feature_names, top_n=15):
    """
    Визуализирует важность признаков для переданной модели-пайплайна.
    model: Pipeline со шагом 'clf' имеющим attribute feature_importances_
    feature_names: список названий признаков
    top_n: количество отображаемых топ-признаков
    """
    import numpy as np

    # Получаем значения важностей
    importances = model.named_steps['clf'].feature_importances_
    # Ищем индексы топ-признаков
    indices = np.argsort(importances)[-top_n:]
    names = [feature_names[i] for i in indices]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), names)
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Feature Importances ({model.named_steps['clf'].__class__.__name__})")
    plt.tight_layout()
    plt.show()
