from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, precision_score, roc_auc_score

def metrics(y_true, y_pred):
    print(f"Accuracy score: {accuracy_score(y_true, y_pred)}")
    print(f"recall score: {recall_score(y_true, y_pred)}")
    print(f"precision score: {precision_score(y_true, y_pred)}")
    print(f"f1 score: {f1_score(y_true, y_pred)}")
    print(f"Classification Report\n{classification_report(y_true, y_pred)}")