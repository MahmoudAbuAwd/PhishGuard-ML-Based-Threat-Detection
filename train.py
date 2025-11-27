import argparse
from src.phishing_pipeline import train_and_save

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='data/phishing.csv')
    parser.add_argument('--model-path', default='models/best_model.pkl')
    args = parser.parse_args()
    best_name, best_metrics, all_results = train_and_save(args.data_path, args.model_path)
    print(f"Best model: {best_name}")
    print(f"Accuracy: {best_metrics['accuracy']:.3f}")
    print(f"F1: {best_metrics['f1']:.3f}")
    print(f"Recall: {best_metrics['recall']:.3f}")
    print(f"Precision: {best_metrics['precision']:.3f}")
    print(best_metrics['report'])

if __name__ == '__main__':
    main()

