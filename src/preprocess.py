import re
import pandas as pd


def preprocess_dataframe(df):
    """
    Applies preprocessing to business_name and business_address.
    Uses vectorized pandas str operations instead of .apply() to avoid
    OOM on 10M-row target tables on 16GB RAM machines.
    """
    print("Preprocessing dataframe...")

    if 'business_name' in df.columns:
        s = df['business_name'].fillna('').astype(str).str.lower()
        s = s.str.replace(r'https?://\S+|www\.\S+', '', regex=True)
        s = s.str.replace(r'\.(com|org|net|co|us|in)\b', '', regex=True)
        s = s.str.replace(r'[,.\-_()"\':|]', ' ', regex=True)
        s = s.str.replace(r'\s+', ' ', regex=True).str.strip()

        # Legal suffix normalization
        suffix_map = {
            r'\bpvt\b': 'private',
            r'\bltd\b': 'limited',
            r'\binc\b': 'incorporated',
            r'\bcorp\b': 'corporation',
            r'\bllc\b': 'llc',
            r'\bllp\b': 'llp',
            r'\bsarl\b': 'sarl',
            r'\bsas\b': 'sas',
        }
        for pattern, replacement in suffix_map.items():
            s = s.str.replace(pattern, replacement, regex=True)

        df['clean_name'] = s.str.replace(r'\s+', ' ', regex=True).str.strip()

    if 'business_address' in df.columns:
        s = df['business_address'].fillna('').astype(str).str.lower()
        s = s.str.replace(r'https?://\S+|www\.\S+', '', regex=True)
        s = s.str.replace(r'\.(com|org|net|co|us|in)\b', '', regex=True)
        s = s.str.replace(r'[,.\-_()"\':|]', ' ', regex=True)
        df['clean_address'] = s.str.replace(r'\s+', ' ', regex=True).str.strip()

    if 'country' in df.columns:
        df['clean_country'] = df['country'].fillna('').astype(str).str.strip().str.lower()

    return df
