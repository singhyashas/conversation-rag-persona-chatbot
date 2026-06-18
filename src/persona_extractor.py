import re
from collections import Counter, defaultdict
from statistics import mean

from .config import PERSONA_MAX_EVIDENCE_PER_ITEM, PERSONA_MAX_ITEMS_PER_CATEGORY
from .text_utils import normalize_space, tokenize


FACT_PATTERNS = (
    re.compile(r"\b(?:i am|i'm)\s+(?:a|an)\s+([^.!?]{3,90})", re.I),
    re.compile(r"\b(?:i work as|i work at|i'm working as|i am working as)\s+([^.!?]{3,90})", re.I),
    re.compile(r"\b(?:i study|i'm studying|i am studying)\s+([^.!?]{3,90})", re.I),
    re.compile(r"\b(?:i live in|i'm from|i am from|i'm originally from|i moved to|i'm moving to)\s+([^.!?]{3,90})", re.I),
    re.compile(r"\bmy\s+([^.!?]{3,40}?)\s+(?:is|are|was|were|works|lives|likes|loves|has|have)\s+([^.!?]{3,70})", re.I),
    re.compile(r"\b(?:i have|i've got)\s+([^.!?]{3,90})", re.I),
)

HABIT_PATTERNS = (
    re.compile(r"\b(?:i usually|i always|i often|i sometimes|i try to|i tend to|i wake up|i spend time|i like to|i love to|i enjoy)\s+([^.!?]{3,90})", re.I),
    re.compile(r"\b(?:in my spare time|for fun)\s*,?\s*(?:i\s+)?([^.!?]{3,90})", re.I),
)

PREFERENCE_PATTERNS = (
    re.compile(r"\b(?:i love|i like|i enjoy|i'm into|i am into|i'm a fan of|i am a fan of)\s+([^.!?]{3,90})", re.I),
    re.compile(r"\bmy favorite\s+(?:thing|book|music|band|meal|food|hobby|activity|song|movie|show)\s+(?:is|are)\s+([^.!?]{3,90})", re.I),
)

TRAIT_SIGNALS = {
    "supportive": ("great job", "you will do great", "happy for you", "encouragement", "i'm sure you"),
    "curious": ("what do you", "what are", "what kind", "favorite", "tell me"),
    "enthusiastic": ("awesome", "amazing", "excited", "love", "so cool"),
    "empathetic": ("sorry to hear", "i understand", "can imagine", "that must be", "i appreciate"),
    "ambitious": ("dream", "college", "career", "job", "study", "becoming", "pursuing"),
    "family-oriented": ("family", "kids", "children", "parents", "mom", "dad", "brother", "sister"),
}

GENERIC_FRAGMENTS = {
    "a lot",
    "fun",
    "good",
    "great",
    "myself",
    "new things",
    "people",
    "that",
    "there",
    "this",
}


def evidence_from_message(message: dict) -> dict:
    return {
        "message_id": message["global_message_id"],
        "conversation_id": message["conversation_id"],
        "speaker": message["speaker"],
        "text": message["text"],
    }


def clean_claim_fragment(fragment: str) -> str:
    fragment = normalize_space(fragment)
    fragment = re.sub(r"^(?:and|but|so|also)\s+", "", fragment, flags=re.I)
    fragment = re.sub(r"^to\s+", "", fragment, flags=re.I)
    fragment = re.sub(r"\s+(?:and|but|so|too|as well|myself)\s*$", "", fragment, flags=re.I)
    return fragment.strip(" ,;:-")


def add_claim(bucket: dict, category: str, label: str, message: dict, claim_type: str) -> None:
    label = clean_claim_fragment(label)
    if len(label) < 3 or len(label.split()) > 16:
        return
    if label.lower() in GENERIC_FRAGMENTS:
        return
    if label.lower().startswith(("and ", "but ", "so ")):
        return

    normalized_label = label.lower()
    item = bucket[category].setdefault(
        normalized_label,
        {
            "claim": label,
            "type": claim_type,
            "count": 0,
            "confidence": "low",
            "evidence": [],
        },
    )
    item["count"] += 1
    if len(item["evidence"]) < PERSONA_MAX_EVIDENCE_PER_ITEM:
        item["evidence"].append(evidence_from_message(message))


def extract_pattern_claims(messages: list[dict]) -> dict:
    by_speaker = defaultdict(lambda: defaultdict(dict))

    for message in messages:
        speaker_bucket = by_speaker[message["speaker"]]
        text = message["text"]

        for pattern in FACT_PATTERNS:
            for match in pattern.finditer(text):
                if len(match.groups()) == 2:
                    label = f"my {match.group(1)} {match.group(2)}"
                else:
                    label = match.group(1)
                add_claim(speaker_bucket, "personal_facts", label, message, "first_person_fact")

        for pattern in HABIT_PATTERNS:
            for match in pattern.finditer(text):
                add_claim(speaker_bucket, "habits", match.group(1), message, "habit_or_routine")

        for pattern in PREFERENCE_PATTERNS:
            for match in pattern.finditer(text):
                add_claim(speaker_bucket, "preferences", match.group(1), message, "stated_preference")

    return by_speaker


def confidence_for_count(count: int) -> str:
    if count >= 3:
        return "high"
    if count == 2:
        return "medium"
    return "low"


def finalize_items(items_by_key: dict) -> list[dict]:
    items = []
    for item in items_by_key.values():
        item["confidence"] = confidence_for_count(item["count"])
        items.append(item)
    items.sort(key=lambda item: (item["count"], len(item["evidence"])), reverse=True)
    return items[:PERSONA_MAX_ITEMS_PER_CATEGORY]


def extract_traits(messages: list[dict]) -> dict:
    trait_buckets = defaultdict(lambda: defaultdict(dict))

    for speaker in sorted({message["speaker"] for message in messages}):
        speaker_messages = [message for message in messages if message["speaker"] == speaker]
        for trait, signals in TRAIT_SIGNALS.items():
            evidence = []
            count = 0
            for message in speaker_messages:
                lowered = message["text"].lower()
                if any(signal in lowered for signal in signals):
                    count += 1
                    if len(evidence) < PERSONA_MAX_EVIDENCE_PER_ITEM:
                        evidence.append(evidence_from_message(message))

            if count:
                trait_buckets[speaker]["personality_traits"][trait] = {
                    "claim": trait,
                    "type": "repeated_language_signal",
                    "count": count,
                    "confidence": confidence_for_count(count),
                    "evidence": evidence,
                }

    return trait_buckets


def communication_style(messages: list[dict]) -> dict:
    output = {}
    for speaker in sorted({message["speaker"] for message in messages}):
        speaker_messages = [message for message in messages if message["speaker"] == speaker]
        texts = [message["text"] for message in speaker_messages]
        token_counts = [len(tokenize(text)) for text in texts]
        question_count = sum("?" in text for text in texts)
        exclamation_count = sum("!" in text for text in texts)
        emoji_count = sum(len(re.findall(r"[\U0001F300-\U0001FAFF]", text)) for text in texts)
        short_count = sum(len(text.split()) <= 8 for text in texts)
        first_person_count = sum(bool(re.search(r"\b(i|i'm|i am|my|me)\b", text, re.I)) for text in texts)

        total = max(len(texts), 1)
        top_terms = Counter()
        for text in texts:
            top_terms.update(tokenize(text))

        style_notes = []
        if question_count / total > 0.22:
            style_notes.append("asks frequent questions")
        if exclamation_count / total > 0.20:
            style_notes.append("uses enthusiastic punctuation")
        if short_count / total > 0.35:
            style_notes.append("often replies briefly")
        if first_person_count / total > 0.25:
            style_notes.append("often shares first-person details")
        if emoji_count:
            style_notes.append("uses emoji")
        if not style_notes:
            style_notes.append("uses a balanced conversational style")

        output[speaker] = {
            "message_count": len(texts),
            "average_content_words": round(mean(token_counts), 2) if token_counts else 0,
            "question_rate": round(question_count / total, 3),
            "exclamation_rate": round(exclamation_count / total, 3),
            "short_message_rate": round(short_count / total, 3),
            "first_person_rate": round(first_person_count / total, 3),
            "emoji_count": emoji_count,
            "style_notes": style_notes,
            "top_terms": [word for word, _ in top_terms.most_common(12)],
        }
    return output


def merge_category(base: dict, incoming: dict) -> None:
    for speaker, categories in incoming.items():
        for category, items in categories.items():
            base[speaker][category].update(items)


def extract_persona(messages: list[dict]) -> dict:
    raw_by_speaker = extract_pattern_claims(messages)
    merge_category(raw_by_speaker, extract_traits(messages))

    speaker_personas = {}
    styles = communication_style(messages)
    for speaker in sorted({message["speaker"] for message in messages}):
        categories = raw_by_speaker[speaker]
        speaker_personas[speaker] = {
            "personal_facts": finalize_items(categories.get("personal_facts", {})),
            "habits": finalize_items(categories.get("habits", {})),
            "preferences": finalize_items(categories.get("preferences", {})),
            "personality_traits": finalize_items(categories.get("personality_traits", {})),
            "communication_style": styles[speaker],
        }

    return {
        "schema_version": "1.0",
        "source": "conversations.csv",
        "method": {
            "description": "Rule-based extraction from chronological first-person messages. Every fact, habit, preference, and trait includes message evidence.",
            "no_guessing_policy": "Claims are included only when matched by explicit text patterns or repeated observable language signals.",
        },
        "speakers": speaker_personas,
    }
