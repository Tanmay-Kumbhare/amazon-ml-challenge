import re
import pandas as pd

def clean_text(text):
    """
    Gentle cleaning:
    - Lowercase
    - Remove multiple spaces
    - Remove generic punctuation but keep alphanumeric and essential characters
    """
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text).lower()
    
    # Strip URLs and domains if they exist (common in source 3)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove .com, .org, etc. at the end of words
    text = re.sub(r'\.(com|org|net|co|us|in)\b', '', text)
    
    # Replace common punctuation with spaces
    text = re.sub(r'[\,\.\-\_\(\)\"\'\:\;\|]', ' ', text)
    
    # Remove excessive spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_legal_suffixes(text):
    """
    Standardize common legal suffixes to avoid false negatives.
    """
    if not text:
        return text
        
    # Map of suffixes
    suffixes = {
        r'\bpvt\b': 'private',
        r'\bltd\b': 'limited',
        r'\binc\b': 'incorporated',
        r'\bcorp\b': 'corporation',
        r'\bllc\b': 'llc',
        r'\bllp\b': 'llp',
        r'\bsarl\b': 'sarl',
        r'\bsas\b': 'sas'
    }
    
    for pattern, replacement in suffixes.items():
        text = re.sub(pattern, replacement, text)
        
    # Remove excessive spaces again
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_dataframe(df):
    """
    Applies preprocessing to business_name and business_address.
    """
    print("Preprocessing dataframe...")
    if 'business_name' in df.columns:
        df['clean_name'] = df['business_name'].apply(clean_text).apply(normalize_legal_suffixes)
    if 'business_address' in df.columns:
        df['clean_address'] = df['business_address'].apply(clean_text)
    if 'country' in df.columns:
        df['clean_country'] = df['country'].apply(lambda x: str(x).strip().lower() if pd.notna(x) else "")
    return df
