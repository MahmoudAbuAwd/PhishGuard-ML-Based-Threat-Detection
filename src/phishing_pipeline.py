import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
import joblib
from .data_processing import load_dataset

def train_models(X_train, y_train):
    models = {}
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)
    models['Logistic Regression'] = log_model

    best_knn = None
    best_knn_score = -1.0
    for k in range(1, 20):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        score = knn.score(X_train, y_train)
        if score > best_knn_score:
            best_knn_score = score
            best_knn = knn
    models['K-Nearest Neighbors'] = best_knn

    svc_grid = GridSearchCV(SVC(), {'gamma': [0.1], 'kernel': ['rbf', 'linear']})
    svc_grid.fit(X_train, y_train)
    models['Support Vector Machine'] = svc_grid
    return models

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'report': classification_report(y_test, y_pred)
        }
        results[name] = metrics
    return results

def select_best_model(models, results):
    best_name = max(results.keys(), key=lambda n: results[n]['f1'])
    return best_name, models[best_name], results[best_name]

def train_and_save(csv_path: str, model_out_path: str):
    X, y = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    best_name, best_model, best_metrics = select_best_model(models, results)
    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    joblib.dump(best_model, model_out_path)
    return best_name, best_metrics, results
