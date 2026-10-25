"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

from grader_contracts.ensembles import EnsembleInput, EnsembleResult

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def model_init(
        random_seed: int = 42,
        max_depth: int = 8,
        min_samples_leaf: int = 5,
) -> RandomForestClassifier:
    """Необученный random forest
    Параметры можно менять в экспериментах в run_experiments"""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_seed,
        n_jobs=-1,
    )


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Финальная проверка выбранной модели: обучение на train и метрики на test.

    Разрешены Random Forest, Extra Trees, AdaBoost, Gradient Boosting,
    HistGradientBoosting, Voting/Stacking или полносвязная MLP. Не используйте
    test_target при обучении модели или подборе её параметров.
    """

    x_train = np.asarray(data.train_features)
    x_test = np.asarray(data.test_features)

    # обучение только на train
    model = model_init(random_seed=data.random_seed)
    model.fit(x_train, data.train_target)

    # предсказания на отложенном test
    proba = model.predict_proba(x_test)[:, 1]  # второй столбец - вероятность класса 1
    pred = (proba >= 0.5).astype(int)

    # метрики на test
    y_test = np.asarray(data.test_target)
    return EnsembleResult(
        model_name="RandomForest",
        probabilities=proba,
        predictions=pred,
        roc_auc=float(roc_auc_score(y_test, proba)),
        accuracy=float(accuracy_score(y_test, pred)),
        precision=float(precision_score(y_test, pred, zero_division=0)),
        recall=float(recall_score(y_test, pred, zero_division=0)),
        f1=float(f1_score(y_test, pred, zero_division=0)),
    )
