import pandas as pd

FEATURES = [
    'PrefixSuffix-',
    'SubDomains',
    'HTTPS',
    'AnchorURL',
    'WebsiteTraffic'
]

def load_raw(csv_path: str):
    return pd.read_csv(csv_path)

def load_dataset(csv_path: str):
    df = load_raw(csv_path)
    if 'Index' in df.columns:
        df = df.drop(['Index'], axis=1)
    X = df[FEATURES]
    y = df['class']
    return X, y

def save_processed(input_csv: str, output_csv: str):
    df = load_raw(input_csv)
    if 'Index' in df.columns:
        df = df.drop(['Index'], axis=1)
    cols = FEATURES + ['class']
    df[cols].to_csv(output_csv, index=False)

