import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

INPUT_FILE = "Sentiment140.csv"
OUTPUT_FILE = "cleaned_sentiment_data.csv"
TEXT_COLUMN = "text"
LABEL_COLUMN = "sentiment"
BATCH_SIZE = 32
CONFIDENCE_THRESHOLD = 0.7
CSV_ENCODING = "latin-1"
ON_BAD_LINES = "skip"

def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), tokenizer, device

def predict_sentiment_batch(texts, model, tokenizer, device, batch_size=BATCH_SIZE):
    preds, confs = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        preds.extend(pred.cpu().tolist())
        confs.extend(conf.cpu().tolist())
    return preds, confs

def preprocess_sentiment140(df):
    df = df.iloc[:, :6]
    df.columns = ['sentiment', 'id', 'date', 'flag', 'user', 'text']
    if 4 in df['sentiment'].unique():
        df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})
        print("Mapped labels 0 and 4 to 0 and 1")
    return df

def clean_dataset(df, model, tokenizer, device,
                  text_col, label_col,
                  batch_size, conf_thresh):
    print(f"Cleaning {len(df)} entries...")
    preds, confs = predict_sentiment_batch(
        df[text_col].tolist(),
        model, tokenizer, device, batch_size
    )
    df['predicted_sentiment'] = preds
    df['prediction_confidence'] = confs

    orig_labels = sorted(df[label_col].unique())
    if set(orig_labels) == {0, 1}:
        mapping = {0: 0, 1: 2}
    elif set(orig_labels) == {-1, 1}:
        mapping = {-1: 0, 1: 2}
    else:
        mapping = None

    if mapping:
        df['original_mapped'] = df[label_col].map(mapping)
    else:
        df['original_mapped'] = df[label_col]

    df['cleaned_sentiment'] = df['original_mapped']

    neutral_mask = (
        (df['predicted_sentiment'] == 1) &
        (df['prediction_confidence'] >= conf_thresh)
    )
    df.loc[neutral_mask, 'cleaned_sentiment'] = 1

    posneg_mask = (
        ~neutral_mask &
        (df['predicted_sentiment'].isin([0, 2])) &
        (df['prediction_confidence'] >= conf_thresh)
    )
    df.loc[posneg_mask, 'cleaned_sentiment'] = df.loc[posneg_mask, 'predicted_sentiment']

    if mapping:
        inv_map = {v: k for k, v in mapping.items()}
        df['final_report_label'] = df['cleaned_sentiment'].map(lambda x: inv_map.get(x, 1))
        changes = (df[label_col] != df['final_report_label']).sum()
        percent = changes / len(df) * 100
        print(f"Relabeled {changes} entries ({percent:.2f}%)")
    else:
        print("Cleaned into three categories without binary mapping report")

    result = df[[text_col, 'cleaned_sentiment']].rename(columns={'cleaned_sentiment': label_col})
    return result, df

def main():
    model, tokenizer, device = load_model()

    try:
        df = pd.read_csv(
            INPUT_FILE,
            header=None,
            encoding=CSV_ENCODING,
            on_bad_lines=ON_BAD_LINES
        )
    except Exception as e:
        print(f"Read error: {e}\nRetrying with Python engine")
        df = pd.read_csv(
            INPUT_FILE,
            header=None,
            encoding=CSV_ENCODING,
            engine='python',
            on_bad_lines=ON_BAD_LINES
        )

    df = preprocess_sentiment140(df)
    df = df.dropna(subset=[TEXT_COLUMN]).reset_index(drop=True)

    cleaned_df, detailed_df = clean_dataset(
        df, model, tokenizer, device,
        TEXT_COLUMN, LABEL_COLUMN,
        BATCH_SIZE, CONFIDENCE_THRESHOLD
    )

    cleaned_df.to_csv(OUTPUT_FILE, index=False)
    detailed_df.to_csv(OUTPUT_FILE.replace('.csv', '_detailed.csv'), index=False)
    print(f"Wrote {OUTPUT_FILE} and detailed output")

if __name__ == "__main__":
    main()
