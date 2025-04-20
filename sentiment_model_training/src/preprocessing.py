import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import config

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip().lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def preprocess_data(path):
    # df = pd.read_csv(path, usecols=['text', 'new_label'])
    df = pd.read_csv(path, usecols=['text', 'label'])
    df.columns = ['text', 'label']
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df = df[:20] #delete
    df['text'] = df['text'].apply(clean_text)
    
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    
    return df, label_encoder

if __name__ == "__main__":
    df, label_encoder = preprocess_data(config.DATA_PATH)
    print(df.head())
    print(label_encoder.classes_)