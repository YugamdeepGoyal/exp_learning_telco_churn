import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
)


def metrics(y_true, y_pred, y_proba):
    print(f"Accuracy score: {accuracy_score(y_true, y_pred)}")
    print(f"recall score: {recall_score(y_true, y_pred)}")
    print(f"precision score: {precision_score(y_true, y_pred)}")
    print(f"f1 score: {f1_score(y_true, y_pred)}")
    print(f"ROC-AUC score: {roc_auc_score(y_true, y_proba)}")
    print(f"Classification Report\n{classification_report(y_true, y_pred)}")
    cm = confusion_matrix(y_true, y_pred)
    plt.title("Confusion Matrix")
    sns.heatmap(
        cm,
        cmap="crest",
        annot=True,
        fmt="d",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        linecolor="white",
        linewidths=0.1
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
