import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import hnswlib

EMBEDDING_DIM = 768


def encode_sentences(
    sentences: list[str],
    model_name: str = "all-mpnet-base-v2",
    batch_size: int = 64,
    device: str = "cuda"
) -> np.ndarray:
    """
    Вычисляет L2-нормализованные эмбеддинги для списка предложений через SBERT.
    Вернёт np.ndarray формы (N, D).
    """
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    # L2-нормализация, чтобы cosine-distance работал корректно
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-12)
    return embeddings

def build_hnsw_index(
    embeddings: np.ndarray,
    mapping: list[str],
    index_path: str,
    space: str = 'cosine',
    ef_construction: int = 200,
    M: int = 16
):
    """
    Строит hnswlib-индекс и сохраняет его + mapping:
      - embeddings: np.ndarray (N, D)
      - mapping: список из N предложений
      - index_path: базовый путь, под который создадутся
          index_path + '.bin'  (сам индекс)
          index_path + '_map.pkl' (pickle с mapping)
    Параметры:
      ef_construction, M — стандартные гиперпараметры HNSW.
    """
    N, D = embeddings.shape
    # инициализируем индекс
    index = hnswlib.Index(space=space, dim=D)
    index.init_index(max_elements=N, ef_construction=ef_construction, M=M)
    # добавляем векторы с id от 0 до N-1
    index.add_items(embeddings, ids=list(range(N)))
    # настройка качества поиска на query time
    index.set_ef(50)

    # сохраняем на диск
    index.save_index(index_path + '.bin')
    with open(index_path + '_map.pkl', 'wb') as f:
        pickle.dump(mapping, f)


def load_hnsw_index(
    index_path: str,
    space: str = 'cosine',
    ef: int = 50
) -> tuple[hnswlib.Index, list[str]]:
    """
    Загружает hnswlib-индекс и соответствующий mapping.
    Возвращает кортеж (index, mapping).
    """
    # 1) Загружаем mapping (список предложений)
    with open(index_path + '_map.pkl', 'rb') as f:
        mapping = pickle.load(f)

    # 2) Инициализируем пустой индекс с правильной размерностью
    index = hnswlib.Index(space=space, dim=EMBEDDING_DIM)
    # 3) Загружаем его из бинарного файла
    index.load_index(index_path + '.bin')
    index.set_ef(ef)
    return index, mapping


def query_hnsw(
    index: hnswlib.Index,
    embeddings: np.ndarray,
    k: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Для батча эмбеддингов возвращает два массива:
      - labels: shape (N, k) — id соседей в mapping
      - distances: shape (N, k) — cosine distance ([0..2], чем меньше, тем ближе)
    """
    labels, distances = index.knn_query(embeddings, k=k)
    return labels, distances