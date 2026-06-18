import math
import re
from collections import Counter


STOPWORDS = {
    "a", "about", "after", "again", "all", "also", "am", "an", "and", "are",
    "as", "at", "be", "been", "but", "by", "can", "could", "did", "do",
    "does", "doing", "for", "from", "get", "go", "going", "good", "got",
    "had", "has", "have", "he", "her", "here", "him", "his", "how", "i",
    "if", "in", "is", "it", "its", "just", "like", "me", "my", "no",
    "not", "of", "on", "or", "our", "really", "she", "so", "some", "that",
    "the", "their", "them", "there", "they", "this", "to", "too", "up",
    "very", "was", "we", "well", "were", "what", "when", "where", "who",
    "will", "with", "would", "yeah", "yes", "you", "your", "thanks", "thank",
    "awesome", "cool", "definitely", "glad", "great", "hear", "hello", "hey",
    "hi", "hope", "know", "maybe", "much", "need", "nice", "oh", "okay",
    "pretty", "should", "sounds", "sure", "think", "try", "want",
}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z']+", text.lower())
        if token not in STOPWORDS and len(token) > 2
    ]


def cosine_similarity(left_tokens: list[str], right_tokens: list[str]) -> float:
    left = Counter(left_tokens)
    right = Counter(right_tokens)
    if not left or not right:
        return 0.0

    shared = set(left) & set(right)
    dot = sum(left[token] * right[token] for token in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def top_keywords(texts: list[str], limit: int = 8) -> list[str]:
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))
    return [word for word, _ in counts.most_common(limit)]
