import json
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def to_lowercase(text):
    return text.lower()


def remove_punctuation(text):
    return re.sub(r'[^a-z0-9\.\!\?]+', ' ', text)


def remove_html_tags(text):
    return re.sub(r'<.*?>', '', text)


def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F" 
                               u"\U0001F300-\U0001F5FF" 
                               u"\U0001F680-\U0001F6FF" 
                               u"\U0001F700-\U0001F77F" 
                               u"\U0001F780-\U0001F7FF" 
                               u"\U0001F800-\U0001F8FF" 
                               u"\U0001F900-\U0001F9FF" 
                               u"\U0001FA00-\U0001FA6F" 
                               u"\U0001FA70-\U0001FAFF" 
                               u"\U00002702-\U000027B0" 
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_urls(text):
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)


def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()


def remove_stopwords(text):
    tokens = text.split()
    return " ".join([word for word in tokens if word not in stop_words])


def full_pipeline(text):
    text = to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_html_tags(text)
    text = remove_emojis(text)
    text = remove_urls(text)
    text = remove_extra_spaces(text)
    #text = remove_stopwords(text)
    return text


def txt_to_json(input_tsv: str, output_json: str):
    df = pd.read_csv(
        input_tsv,
        sep='\t',
        header=None,
        names=['sentence1', 'sentence2', 'label'],
        encoding='utf-8'
    )
    df.dropna(subset=['sentence1', 'sentence2'], inplace=True)
    df.drop_duplicates(subset=['sentence1', 'sentence2', 'label'], inplace=True)
    df['sentence1'] = df['sentence1'].apply(full_pipeline)
    df['sentence2'] = df['sentence2'].apply(full_pipeline)

    records = df.to_dict(orient='records')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(records)} pairs from\n  {input_tsv}\n→ {output_json}")


if __name__ == "__main__":
    txt_to_json(
        input_tsv="D:/PythonProjects/Plagiat/data/train_snli.txt",
        output_json="D:/PythonProjects/Plagiat/data/train_pairs1.json"
    )
