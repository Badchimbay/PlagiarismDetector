import re
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')


def clean_text(text: str) -> str:
    """
    Приводит текст к единому виду:
    - lowercase
    - удаляет лишние пробелы
    - отбрасывает небуквенные символы, кроме пунктуации
    """
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z0-9\.\!\?]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_into_sentences(text: str) -> list[str]:
    """
    Разбивает очищенный текст на предложения.
    Используем NLTK для корректной работы с английскими аббревиатурами.
    """
    cleaned = clean_text(text)
    sentences = sent_tokenize(cleaned)
    sentences = [s for s in sentences if len(s.split()) > 2]
    return sentences