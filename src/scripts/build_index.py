from pathlib import Path
import json

from src.utils.text_cleaning import split_into_sentences
from src.utils.sbert_index import encode_sentences, build_hnsw_index


def main():
    BASE_DIR = Path(__file__).resolve().parents[2]
    raw_dir = BASE_DIR / "data" / "raw"
    print("Looking for .txt files in:", raw_dir.resolve())
    index_base = BASE_DIR / "models" / "hnsw_index"

    sentences = []
    for txt_file in raw_dir.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8")
        sents = split_into_sentences(text)
        sentences.extend(sents)

    sentences = list(dict.fromkeys(sentences))
    print(f"Collected {len(sentences)} unique sentences.")

    print("Encoding sentences with SBERT…")
    embeddings = encode_sentences(sentences, model_name="all-mpnet-base-v2", device="cuda")

    print("Building HNSW index…")
    build_hnsw_index(
        embeddings=embeddings,
        mapping=sentences,
        index_path=str(index_base),
        ef_construction=200,
        M=16
    )
    print(f"Index saved to {index_base}.bin and {index_base}_map.pkl")

    mapping_json = BASE_DIR / "models" / "hnsw_index_mapping.json"
    mapping_json.write_text(json.dumps(sentences, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Mapping also saved to {mapping_json}")


if __name__ == "__main__":
    main()
