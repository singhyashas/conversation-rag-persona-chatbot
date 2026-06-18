import re

from .text_utils import normalize_space, tokenize, top_keywords


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", normalize_space(text))
    return [piece.strip() for piece in pieces if piece.strip()]


def summarize_messages(messages: list[dict], sentence_limit: int = 3) -> str:
    if not messages:
        return ""

    text_by_message = [message["text"] for message in messages]
    full_text = " ".join(text_by_message)
    sentences = split_sentences(full_text)
    if not sentences:
        return normalize_space(full_text[:280])

    keywords = top_keywords(text_by_message, limit=10)
    keyword_set = set(keywords)
    scored = []
    for index, sentence in enumerate(sentences):
        tokens = tokenize(sentence)
        if not tokens:
            score = 0.0
        else:
            keyword_hits = sum(1 for token in tokens if token in keyword_set)
            score = keyword_hits + min(len(tokens), 18) * 0.08
        if re.search(r"\b(i am|i'm|i have|i love|i like|i enjoy|my |work|study|family|job|hobby|moving)\b", sentence, re.I):
            score += 1.0
        if len(tokens) < 4:
            score -= 2.0
        score += max(0, 0.35 - index * 0.01)
        scored.append((score, index, sentence))

    selected = sorted(scored, reverse=True)[:sentence_limit]
    selected_in_order = [sentence for _, _, sentence in sorted(selected, key=lambda item: item[1])]
    summary = " ".join(selected_in_order)
    return normalize_space(summary)


def build_summary_record(messages: list[dict], extra: dict | None = None) -> dict:
    record = {
        "start_message_id": messages[0]["global_message_id"],
        "end_message_id": messages[-1]["global_message_id"],
        "message_count": len(messages),
        "keywords": top_keywords([message["text"] for message in messages]),
        "summary": summarize_messages(messages),
    }
    if extra:
        record.update(extra)
    return record
