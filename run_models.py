"""Перебор конфигураций + запуск контрактной модели. Пишет experiment_results.json."""
import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from grader_contracts.ensembles import EnsembleInput
from ensemble_models import prepare_data, run_ensemble_experiment

SEED = 42

CONFIGS = [
    {"max_depth": 4, "min_samples_leaf": 10},
    {"max_depth": 8, "min_samples_leaf": 10},
    {"max_depth": 12, "min_samples_leaf": 10},
    {"max_depth": 8, "min_samples_leaf": 5},
    {"max_depth": 8, "min_samples_leaf": 20},
]


def metrics(target_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": round(roc_auc_score(target_true, probabilities), 4),
        "accuracy": round(accuracy_score(target_true, predictions), 4),
        "precision": round(precision_score(target_true, predictions, zero_division=0), 4),
        "recall": round(recall_score(target_true, predictions, zero_division=0), 4),
        "f1": round(f1_score(target_true, predictions, zero_division=0), 4),
    }


def run_all_configs(features_train, features_test, target_train, target_test):
    """Перебирает CONFIGS.
    Возвращает строки с параметрами и метриками"""
    prepared_train, prepared_test = prepare_data(features_train, features_test)
    experiments = []
    for config in CONFIGS:
        model = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, **config)
        model.fit(prepared_train, target_train)
        probabilities = model.predict_proba(prepared_test)[:, 1]
        experiments.append({**config, **metrics(target_test, probabilities)})
    return experiments


def run_contract_model(features_train, features_test, target_train, target_test):
    """Запускает контрактную модель из model_init()"""
    data = EnsembleInput(features_train, target_train, features_test, target_test, SEED)
    return run_ensemble_experiment(data)


def save_results(contract_result, experiments):
    """Логирует метрики контрактной модели + история перебора CONFIGS"""
    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump({
            # метрики контрактной модели (их ждёт проверка)
            "model_name": contract_result.model_name,
            "roc_auc": contract_result.roc_auc,
            "accuracy": contract_result.accuracy,
            "precision": contract_result.precision,
            "recall": contract_result.recall,
            "f1": contract_result.f1,
            "random_seed": SEED,
            "split": {"test_size": 0.2, "stratify": True, "random_state": SEED},
            # история перебора
            "experiments": experiments,
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    df = pd.read_csv("./german.csv", sep=";")
    target = df["Creditability"]
    features = df.drop(columns=["Creditability"])

    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.2, random_state=SEED, stratify=target
    )

    experiments = run_all_configs(features_train, features_test, target_train, target_test)
    contract_result = run_contract_model(features_train, features_test, target_train, target_test)
    save_results(contract_result, experiments)
