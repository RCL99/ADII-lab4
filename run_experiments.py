"""Сравнивает три конфига Random Forest по CV на train и логирует experiment_results.json"""
import json

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from grader_contracts.ensembles import EnsembleInput
from ensemble_models import model_init, run_ensemble_experiment

SEED = 42

CONFIGS = [
    ("RandomForest depth=4 leaf=10", {"max_depth": 4, "min_samples_leaf": 10}),  # простой конфиг
    ("RandomForest depth=8 leaf=5", {"max_depth": 8, "min_samples_leaf": 5}),  # выбранный конфиг
    ("RandomForest depth=12 leaf=10", {"max_depth": 12, "min_samples_leaf": 10}),  # сложный конфиг
]


def run_experiments(features_train, target_train):
    """Считает метрики конфигов по CV только на train"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = ["roc_auc", "accuracy", "precision", "recall", "f1"]

    rows = []
    for name, params in CONFIGS:
        model = model_init(SEED, **params)
        scores = cross_validate(model, features_train, target_train, cv=cv, scoring=scoring)
        row = {"model": name}
        for metric in scoring:
            row[f"cv_{metric}"] = round(float(scores[f"test_{metric}"].mean()), 4)
        rows.append(row)
    return rows


def save_results(contract_result, experiments):
    """Пишет метрики финальной проверки и историю перебора конфигов по CV"""
    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump({
            # метрики финальной проверки на отложенном test
            "model_name": contract_result.model_name,
            "roc_auc": contract_result.roc_auc,
            "accuracy": contract_result.accuracy,
            "precision": contract_result.precision,
            "recall": contract_result.recall,
            "f1": contract_result.f1,
            "random_seed": SEED,
            "split": {"test_size": 0.2, "stratify": True, "random_state": SEED},
            # история перебора конфигов по CV
            "experiments": experiments,
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print("Start")

    df = pd.read_csv("./german.csv", sep=";")
    target = df["Creditability"]
    features = df.drop(columns=["Creditability"])

    # тест на 20% строк
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.2, random_state=SEED, stratify=target
    )

    # перебор конфигов
    experiments = run_experiments(features_train, target_train)

    # финальная проверка run_ensemble_experiment
    data = EnsembleInput(features_train, target_train, features_test, target_test, SEED)
    contract_result = run_ensemble_experiment(data)
    save_results(contract_result, experiments)

    print("Finish")
