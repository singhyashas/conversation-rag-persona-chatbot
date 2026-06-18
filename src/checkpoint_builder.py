from .config import (
    MIN_TOPIC_MESSAGES,
    SUMMARY_SENTENCE_LIMIT,
    TOPIC_SIMILARITY_THRESHOLD,
    TOPIC_WINDOW_SIZE,
)
from .summarizer import build_summary_record
from .text_utils import cosine_similarity, normalize_space, tokenize


TOPIC_SHIFT_CUES = (
    "anyway",
    "by the way",
    "speaking of",
    "what else",
    "so what",
    "what do you do",
    "what about",
    "for fun",
    "your hobbies",
    "your job",
    "do you have any",
)


def _recent_topic_text(messages: list[dict]) -> str:
    return " ".join(message["text"] for message in messages[-TOPIC_WINDOW_SIZE:])


def _has_shift_cue(text: str) -> bool:
    lowered = text.lower()
    return any(cue in lowered for cue in TOPIC_SHIFT_CUES)


def should_start_new_topic(active_messages: list[dict], message: dict) -> tuple[bool, float, list[str]]:
    if len(active_messages) < 4:
        return False, 1.0, []

    recent_text = _recent_topic_text(active_messages)
    similarity = cosine_similarity(tokenize(recent_text), tokenize(message["text"]))
    current_tokens = tokenize(message["text"])
    reasons = []

    if similarity < TOPIC_SIMILARITY_THRESHOLD and len(active_messages) >= MIN_TOPIC_MESSAGES + 2 and len(current_tokens) >= 5:
        reasons.append(f"low_similarity:{similarity:.3f}")
    if _has_shift_cue(message["text"]):
        reasons.append("topic_shift_cue")

    should_split = bool(reasons) and (
        similarity < TOPIC_SIMILARITY_THRESHOLD or _has_shift_cue(message["text"])
    )
    return should_split, similarity, reasons


def build_topic_checkpoints(messages: list[dict]) -> list[dict]:
    checkpoints = []
    active = []
    topic_id = 1
    split_events = []

    for message in messages:
        if active:
            should_split, similarity, reasons = should_start_new_topic(active, message)
            conversation_changed = message["conversation_id"] != active[-1]["conversation_id"]
            if conversation_changed or should_split:
                reason = ["new_conversation"] if conversation_changed else reasons
                checkpoint = build_summary_record(
                    active,
                    {
                        "topic_id": topic_id,
                        "split_reason": reason,
                    },
                )
                checkpoints.append(checkpoint)
                split_events.append(
                    {
                        "topic_id": topic_id,
                        "next_message_id": message["global_message_id"],
                        "similarity": round(similarity, 4),
                        "reason": reason,
                    }
                )
                topic_id += 1
                active = []

        active.append(message)

    if active:
        checkpoints.append(
            build_summary_record(
                active,
                {
                    "topic_id": topic_id,
                    "split_reason": ["end_of_dataset"],
                },
            )
        )

    return checkpoints


def build_hundred_message_checkpoints(messages: list[dict], size: int = 100) -> list[dict]:
    checkpoints = []
    for index in range(0, len(messages), size):
        chunk = messages[index : index + size]
        checkpoint_id = len(checkpoints) + 1
        checkpoints.append(
            build_summary_record(
                chunk,
                {
                    "checkpoint_id": checkpoint_id,
                    "chunk_size": size,
                },
            )
        )
    return checkpoints


def checkpoint_preview(checkpoints: list[dict], limit: int = 5) -> str:
    lines = []
    for checkpoint in checkpoints[:limit]:
        label = checkpoint.get("topic_id", checkpoint.get("checkpoint_id"))
        lines.append(
            normalize_space(
                f"{label}: messages {checkpoint['start_message_id']}-{checkpoint['end_message_id']} "
                f"({checkpoint['message_count']} msgs) -> {checkpoint['summary']}"
            )
        )
    return "\n".join(lines)
