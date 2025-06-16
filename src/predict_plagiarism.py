from pathlib import Path
import json
from src.utils.text_cleaning import split_into_sentences
from src.utils.tokenization import load_tokenizer, texts_to_padded_sequences
from src.train_model import build_siamese_lstm
from src.utils.sbert_index import (
    encode_sentences,
    load_hnsw_index,
    query_hnsw
)

K = 5         # сколько кандидатов вытягивать из индекса
PROB_THRESHOLD = 0.90      # порог вероятности LSTM-классификатора
DIST_THRESHOLD = 0.30      # максимальная косинусная дистанция
TOP_N = 3         # сколько лучших совпадений вернуть
NUM_WORDS = 50000     # словарь для LSTM
EMBEDDING_DIM = 100       # размер эмбеддингов в LSTM
MAXLEN = 40        # максимальная длина предложения для LSTM

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_BASE = BASE_DIR / "models" / "hnsw_index"
MODEL_PATH = BASE_DIR / "models" / "lstm_model_batch128.h5"
TOKENIZER_PATH = BASE_DIR / "models" / "tokenizer_batch128.pkl"

_index, _mapping = None, None
_lstm_model, _tokenizer = None, None


def _load_resources():
    global _index, _mapping, _lstm_model, _tokenizer

    if _index is None or _mapping is None:
        _index, _mapping = load_hnsw_index(str(INDEX_BASE))

    if _lstm_model is None or _tokenizer is None:
        _lstm_model = build_siamese_lstm(
            vocab_size=NUM_WORDS,
            embedding_dim=EMBEDDING_DIM,
            maxlen=MAXLEN
        )
        _lstm_model.load_weights(str(MODEL_PATH))
        _tokenizer = load_tokenizer(str(TOKENIZER_PATH))

    return _index, _mapping, _lstm_model, _tokenizer


def check_plagiarism(text: str) -> dict:
    """
    Основная функция. Принимает текст, разбивает его на предложения,
    извлекает кандидатов через HNSW, фильтрует их и прогоняет через Siamese-LSTM.
    Возвращает dict с ключами:
      "summary": {
          "total_sentences": int,
          "flagged_sentences": int,
          "plagiarism_percent": float
      },
      "details": [
          {
            "sentence": str,
            "matches": [
               {"source": str, "score": float, "dist": float}, ...
            ]
          }, ...
      ]
    """
    # 1) Разбиение на предложения
    sents = split_into_sentences(text)
    if not sents:
        return {"summary": {"total_sentences": 0,
                             "flagged_sentences": 0,
                             "plagiarism_percent": 0.0},
                "details": []}

    # 2) Загрузка HNSW, модели и токенизатора
    index, mapping, lstm_model, tokenizer = _load_resources()

    # 3) эмбеддинг + топ-K соседей
    embs = encode_sentences(sents, device="cpu")
    labels, distances = query_hnsw(index, embs, k=K)

    report = []
    flagged_count = 0

    # 4) Verification: для каждой пары “предложение – кандидат”
    for i, sent in enumerate(sents):
        cand_ids = labels[i]
        cand_dists = distances[i]

        candidates = [mapping[idx] for idx in cand_ids]

        seq1 = texts_to_padded_sequences(tokenizer, [sent]*K,
                                         maxlen=lstm_model.input_shape[0][1])
        seq2 = texts_to_padded_sequences(tokenizer, candidates,
                                         maxlen=lstm_model.input_shape[0][1])

        probs = lstm_model.predict([seq1, seq2], verbose=0).flatten()

        # Сборка (source, prob, dist) и фильтрация
        triples = list(zip(candidates, probs, cand_dists))
        filtered = [
            (txt, float(p), float(d))
            for txt, p, d in triples
            if p >= PROB_THRESHOLD and d <= DIST_THRESHOLD
        ]

        top_matches = sorted(filtered, key=lambda x: x[1], reverse=True)[:TOP_N]

        matches = [
            {"source": txt, "score": p, "dist": d}
            for txt, p, d in top_matches
        ]

        if matches:
            flagged_count += 1

        report.append({
            "sentence": sent,
            "matches": matches
        })

    # 5) Итоговая статистика
    total = len(sents)
    percent = flagged_count / total * 100

    summary = {
        "total_sentences": total,
        "flagged_sentences": flagged_count,
        "plagiarism_percent": round(percent, 1)
    }

    return {"summary": summary, "details": report}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_txt", type=Path,
                        help="путь к файлу с текстом для проверки")
    parser.add_argument("--output_json", type=Path,
                        default=BASE_DIR / "results" / "plagiarism_report.json")
    args = parser.parse_args()

    text = args.input_txt.read_text(encoding="utf-8")
    result = check_plagiarism(text)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Done. Report saved to {args.output_json}")
