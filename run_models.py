import json
import pandas as pd
from sklearn.model_selection import train_test_split
from grader_contracts.ensembles import EnsembleInput, EnsembleResult
from ensemble_models import run_ensemble_experiment


def output_results(result: EnsembleResult):
    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_name": result.model_name,
            "roc_auc": result.roc_auc,
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    df = pd.read_csv("./german.csv", sep=';')

    y = df["Creditability"]  # цель
    x = df.drop(columns=["Creditability"])  # признаки

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=843, stratify=y
    )

    data = EnsembleInput(
        train_features=x_train,
        train_target=y_train,
        test_features=x_test,
        test_target=y_test,
        random_seed=843,
    )

    result = run_ensemble_experiment(data)
    output_results(result)
