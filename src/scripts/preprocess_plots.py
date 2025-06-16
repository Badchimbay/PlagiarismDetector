import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from src.scripts.convert_snli import full_pipeline

BASE_DIR = Path(__file__).resolve().parent.parent.parent
processed_json = BASE_DIR / "data" / "train_pairs1.json"
raw_tsv = BASE_DIR / "data" / "train_snli.txt"
proc_json = processed_json

df_raw = pd.read_csv(raw_tsv, sep="\t", header=None,
                      names=["sentence1", "sentence2", "label"]).head()
df_proc = pd.read_json(proc_json)[["sentence1", "sentence2", "label"]].head()

print("Таблица 4.1. Пример исходных и обработанных данных:")
print("\nИсходный TSV:")
print(df_raw.to_string(index=False))
print("\nПосле предобработки (JSON):")
print(df_proc.to_string(index=False))

df_raw = pd.read_csv(
    raw_tsv, sep="\t", header=None,
    names=["sentence1", "sentence2", "label"],
    encoding="utf-8"
)
df_raw.dropna(subset=['sentence1', 'sentence2'], inplace=True)
df_raw.drop_duplicates(subset=['sentence1', 'sentence2', 'label'], inplace=True)

sentences_raw = df_raw["sentence1"].tolist() + df_raw["sentence2"].tolist()
lengths_raw = [len(s.split()) for s in sentences_raw]
lengths_clean = [len(full_pipeline(s).split()) for s in sentences_raw]

max_len = 60

plt.figure()
plt.hist([l for l in lengths_raw   if l <= max_len], bins=range(max_len+1), alpha=0.5, color='#009999', label="До очистки")
plt.hist([l for l in lengths_clean if l <= max_len], bins=range(max_len+1), alpha=0.5, color='#ff7400', label="После очистки")
plt.xlabel("Количество токенов в предложении")
plt.ylabel("Частота")
plt.title("Распределение длин предложений\nдо и после предобработки")
plt.legend()
plt.tight_layout()
plt.show()