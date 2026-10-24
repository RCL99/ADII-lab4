import pandas as pd
from sklearn.model_selection import train_test_split
from grader_contracts.ensembles import EnsembleInput
from ensemble_models import run_ensemble_experiment

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

    print(f"model:    {result.model_name}")
    print(f"ROC-AUC:  {result.roc_auc:.4f}")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Precision:{result.precision:.4f}")
    print(f"Recall:   {result.recall:.4f}")
    print(f"F1:       {result.f1:.4f}")
