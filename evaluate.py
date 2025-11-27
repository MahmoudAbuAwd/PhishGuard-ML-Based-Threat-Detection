import argparse
import joblib
from src.phishing_pipeline import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='data/phishing.csv')
    parser.add_argument('--model-path', default='models/best_model.pkl')
    args = parser.parse_args()
    X, y = load_dataset(args.data_path)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = joblib.load(args.model_path)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"F1: {f1_score(y_test, y_pred):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()

