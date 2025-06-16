import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def build_tokenizer(texts, num_words=50000, oov_token="<OOV>"):
    """
    Строит и обучает Keras Tokenizer на списке текстов.
    texts: list[str] — корпус для fit_on_texts()
    num_words: максимальный размер словаря
    oov_token: токен для неизвестных слов
    """
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(texts)
    return tokenizer


def save_tokenizer(tokenizer, filepath):
    """
    Сохраняет объект tokenizer в файл (pickle).
    """
    with open(filepath, "wb") as f:
        pickle.dump(tokenizer, f)


def load_tokenizer(filepath):
    """
    Загружает tokenizer из файла (pickle).
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def texts_to_padded_sequences(tokenizer, texts, maxlen):
    """
    Преобразует список строк в список числовых последовательностей + паддинг.
    Возвращает numpy array формы (len(texts), maxlen).
    """
    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(
        seqs,
        maxlen=maxlen,
        padding="post",      # добавляем нули в конец
        truncating="post"    # усекаем лишние токены в конце
    )
    return padded