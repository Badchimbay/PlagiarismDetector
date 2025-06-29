import os.path
from pathlib import Path
import json
import glob
from docx import Document
import pdfplumber

from src.utils.text_cleaning import split_into_sentences
from src.utils.sbert_index import encode_sentences, build_hnsw_index


def build_index():
    BASE_DIR = Path(__file__).resolve().parents[2]
    files = glob.glob(f'{BASE_DIR}/models/hnsw_index*')
    for file in files:
        os.remove(file)
    raw_dir = BASE_DIR / "data" / "raw"
    print("Looking for .txt, .docx, .pdf files in:", raw_dir.resolve())
    index_base = BASE_DIR / "models" / "hnsw_index"

    sentences = []
    for file_path in raw_dir.iterdir():
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            text = file_path.read_text(encoding="utf-8")
        elif suffix == ".docx":
            text = read_docx(file_path)
        elif suffix == ".pdf":
            text = read_pdf(file_path)
        else:
            continue
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


def read_docx(file_path):
    doc = Document(file_path)
    return '\n'.join(para.text for para in doc.paragraphs)


def read_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text