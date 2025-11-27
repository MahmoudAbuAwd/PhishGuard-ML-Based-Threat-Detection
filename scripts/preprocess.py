import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data_processing import save_processed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/phishing.csv')
    parser.add_argument('--output', default='data/processed.csv')
    args = parser.parse_args()
    save_processed(args.input, args.output)
    print(f"Saved processed dataset to {args.output}")

if __name__ == '__main__':
    main()
