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
    roc_curve
)


def metrics(y_true, y_pred, y_proba, model_name=None):
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

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="ROC CURVE")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC AUC Curve")
    plt.legend(loc="lower right")
    plt.show()
