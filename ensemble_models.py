"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

from grader_contracts.ensembles import EnsembleInput, EnsembleResult

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def model_init(random_seed: int = 42) -> RandomForestClassifier:
    """Необученный random forest с параметрами под маленький датасет"""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        random_state=random_seed,
        n_jobs=-1,
    )


def prepare_data(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Заполнение пропусков
    На выходе подготовленные train и test
    """
    cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    # замена пропусков медианой
    num_imp = SimpleImputer(strategy="median")
    x_train = num_imp.fit_transform(X_train[cols])
    x_test = num_imp.transform(X_test[cols])

    return x_train, x_test


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучите выбранную модель и верните вероятности, метки и метрики на test.

    Разрешены Random Forest, Extra Trees, AdaBoost, Gradient Boosting,
    HistGradientBoosting, Voting/Stacking или полносвязная MLP. Не используйте
    test_target при обучении модели или подборе её параметров.
    """

    x_train, x_test = prepare_data(data.train_features, data.test_features)

    # обучение модели на train
    model = model_init(random_seed=data.random_seed)
    model.fit(x_train, data.train_target)

    # предсказания
    proba = model.predict_proba(x_test)[:, 1]  # второй столбец - вероятность класса 1
    pred = np.where(proba >= 0.5, 1, 0)

    # метрики
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
