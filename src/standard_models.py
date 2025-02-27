from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import  f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, auc
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
#from xgboost import XGBClassifier

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def classifier_model_test(X_train, X_test, y_train, y_test):
    classifiers = [
        GradientBoostingClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        ExtraTreesClassifier(),
        DecisionTreeClassifier(),
        KNeighborsClassifier(),
        GaussianNB(),
        BernoulliNB(),
        SVC(probability=True),
        LogisticRegression(),
        SGDClassifier(),
        OneClassSVM(nu=0.01, kernel="rbf", gamma="scale"),
        IsolationForest(contamination=0.01)
    ]
    
    classifier_names = [
        'GradientBoost', 'RandomForest', 'AdaBoost', 'ExtraTrees', 'DecisionTree', 'KNeighbors',
        'GaussianNB', 'BernoulliNB', 'SVC', 'LogisticRegression', 'SGD', 'One-Class SVM', 'Isolation Forest'
    ]

    metrics_df = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-PR'], index=classifier_names)
    confusion_dict = {}
    
    plt.figure(figsize=(10, 6))
    
    for i, clf in enumerate(classifiers):
        name = classifier_names[i]
        
        if name in ['One-Class SVM', 'Isolation Forest']:
            clf.fit(X_train)
            y_pred = clf.predict(X_test)
            y_pred_binary = (y_pred == -1).astype(int)
        else:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_pred_binary = y_pred
            if hasattr(clf, "predict_proba"):
                y_scores = clf.predict_proba(X_test)[:, 1]
            else:
                y_scores = y_pred_binary  # Use binary output if probability is not available
        
        acc = accuracy_score(y_test, y_pred_binary)
        prec = precision_score(y_test, y_pred_binary, zero_division=0)
        rec = recall_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_scores)
        auc_pr = auc(recall_vals, precision_vals)
        
        metrics_df.loc[name] = [acc, prec, rec, f1, auc_pr]
        confusion_dict[name] = confusion_matrix(y_test, y_pred_binary)
        
        plt.plot(recall_vals, precision_vals, label=f'{name} (AUC={auc_pr:.2f})')
    
    metrics_df = metrics_df.sort_values(by='F1', ascending=False)
    
    print("=== CONFUSION MATRICES ===")
    for name in metrics_df.index:
        tn, fp, fn, tp = confusion_dict[name].ravel()
        print(f"{name} Confusion Matrix: TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=metrics_df.index, y='F1', data=metrics_df, palette='Blues_d')
    plt.xticks(rotation=45, ha='right')
    plt.title("F1 Scores by Classifier")
    plt.xlabel("Classifier")
    plt.ylabel("F1 Score")
    plt.show()
    
    return metrics_df