import math
from collections import Counter, defaultdict
from pathlib import Path

from .config import MESSAGE_CHUNK_OVERLAP, MESSAGE_CHUNK_SIZE
from .text_utils import tokenize


def format_message(message: dict) -> str:
    return f"{message['speaker']}: {message['text']}"


def build_message_chunks(
    messages: list[dict],
    chunk_size: int = MESSAGE_CHUNK_SIZE,
    overlap: int = MESSAGE_CHUNK_OVERLAP,
) -> list[dict]:
    chunks = []
    stride = max(1, chunk_size - overlap)
    for start in range(0, len(messages), stride):
        chunk_messages = messages[start : start + chunk_size]
        if not chunk_messages:
            continue
        chunk_id = len(chunks) + 1
        chunks.append(
            {
                "chunk_id": chunk_id,
                "start_message_id": chunk_messages[0]["global_message_id"],
                "end_message_id": chunk_messages[-1]["global_message_id"],
                "message_count": len(chunk_messages),
                "conversation_ids": sorted({message["conversation_id"] for message in chunk_messages}),
                "text": "\n".join(format_message(message) for message in chunk_messages),
            }
        )
    return chunks


def build_retrieval_documents(
    topic_checkpoints: list[dict],
    hundred_checkpoints: list[dict],
    message_chunks: list[dict],
) -> list[dict]:
    documents = []

    for checkpoint in topic_checkpoints:
        documents.append(
            {
                "doc_id": f"topic_{checkpoint['topic_id']}",
                "type": "topic_summary",
                "start_message_id": checkpoint["start_message_id"],
                "end_message_id": checkpoint["end_message_id"],
                "title": f"Topic {checkpoint['topic_id']}",
                "text": (
                    f"Topic summary: {checkpoint['summary']}\n"
                    f"Keywords: {', '.join(checkpoint.get('keywords', []))}"
                ),
                "metadata": {
                    "topic_id": checkpoint["topic_id"],
                    "message_count": checkpoint["message_count"],
                    "split_reason": checkpoint.get("split_reason", []),
                },
            }
        )

    for checkpoint in hundred_checkpoints:
        documents.append(
            {
                "doc_id": f"hundred_{checkpoint['checkpoint_id']}",
                "type": "hundred_message_summary",
                "start_message_id": checkpoint["start_message_id"],
                "end_message_id": checkpoint["end_message_id"],
                "title": f"100-message checkpoint {checkpoint['checkpoint_id']}",
                "text": (
                    f"100-message summary: {checkpoint['summary']}\n"
                    f"Keywords: {', '.join(checkpoint.get('keywords', []))}"
                ),
                "metadata": {
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "message_count": checkpoint["message_count"],
                },
            }
        )

    for chunk in message_chunks:
        documents.append(
            {
                "doc_id": f"chunk_{chunk['chunk_id']}",
                "type": "message_chunk",
                "start_message_id": chunk["start_message_id"],
                "end_message_id": chunk["end_message_id"],
                "title": f"Messages {chunk['start_message_id']}-{chunk['end_message_id']}",
                "text": chunk["text"],
                "metadata": {
                    "chunk_id": chunk["chunk_id"],
                    "message_count": chunk["message_count"],
                    "conversation_ids": chunk["conversation_ids"],
                },
            }
        )

    return documents


class BM25Retriever:
    def __init__(self, documents: list[dict], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_lengths = []
        self.doc_types = []
        self.inverted_index = defaultdict(list)
        self.doc_freq = Counter()
        self.avg_doc_length = 0.0
        self._build_index()

    def _build_index(self) -> None:
        total_length = 0
        for doc_index, document in enumerate(self.documents):
            counts = Counter(tokenize(document["text"]))
            length = sum(counts.values())
            self.doc_lengths.append(length)
            self.doc_types.append(document["type"])
            total_length += length

            for term, frequency in counts.items():
                self.inverted_index[term].append((doc_index, frequency))
                self.doc_freq[term] += 1

        self.avg_doc_length = total_length / max(len(self.documents), 1)

    def _idf(self, term: str) -> float:
        total_docs = len(self.documents)
        docs_with_term = self.doc_freq.get(term, 0)
        return math.log(1 + (total_docs - docs_with_term + 0.5) / (docs_with_term + 0.5))

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_type: str | tuple[str, ...] | None = None,
    ) -> list[dict]:
        query_terms = tokenize(query)
        if not query_terms:
            return []

        allowed_types = None
        if doc_type:
            allowed_types = {doc_type} if isinstance(doc_type, str) else set(doc_type)

        scores = defaultdict(float)
        for term in query_terms:
            idf = self._idf(term)
            for doc_index, frequency in self.inverted_index.get(term, []):
                if allowed_types and self.doc_types[doc_index] not in allowed_types:
                    continue
                doc_length = self.doc_lengths[doc_index]
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1)
                )
                scores[doc_index] += idf * frequency * (self.k1 + 1) / denominator

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        results = []
        for doc_index, score in ranked:
            document = dict(self.documents[doc_index])
            document["score"] = round(score, 4)
            results.append(document)
        return results


def load_documents(path: Path) -> list[dict]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
