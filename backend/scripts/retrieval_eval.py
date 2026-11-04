from __future__ import annotations

import json
import math
import re
import unicodedata


def _normalize_token(token: str) -> str:
    """Strip accents and lowercase tokens so equivalent terms are treated as the same word during evaluation."""
    token = unicodedata.normalize("NFD", token.lower())
    token = token.encode("ascii", "ignore").decode("utf-8")
    return token


def _tokenize(text: str) -> list[str]:
    """Normalize text into lowercase tokens so similarity checks are stable and deterministic."""
    return [_normalize_token(token) for token in re.findall(r"[a-zA-Záéíóúñ]+", text)]


def _build_vocabulary() -> list[str]:
    """Define the compact vocabulary for the evaluation corpus so the synthetic tests stay reproducible."""
    return [
        "poliza",
        "cubre",
        "danos",
        "materiales",
        "vehiculo",
        "asistencia",
        "carretera",
        "robo",
        "colision",
        "accidente",
        "exclusion",
        "exclusiones",
        "muerte",
        "salud",
        "hospital",
        "hospitalarios",
        "deducible",
        "reembolso",
        "garantia",
        "gastos",
        "dental",
        "medica",
        "medicos",
        "urgencias",
        "vigencia",
        "seguro",
        "culpa",
        "clausula",
        "riesgo",
        "cobertura",
        "cotizacion",
        "urgencia",
    ]


def _embedding_from_text(text: str, vocabulary: list[str]) -> list[float]:
    """Create a simple deterministic BoW-style vector from the text to mimic retrieval relevance without external dependencies."""
    counts = {word: 0 for word in vocabulary}
    for token in _tokenize(text):
        if token in counts:
            counts[token] += 1
        elif token.endswith("s") and token[:-1] in counts:
            counts[token[:-1]] += 1
    return [counts[word] for word in vocabulary]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Measure vector similarity using cosine distance; zero vectors yield a neutral score of zero."""
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def _matches_expected_topic(expected_topic: str, text: str) -> bool:
    """Check topic terms independent of accents, word order, and Spanish connector words."""
    stopwords = {"a", "con", "de", "del", "el", "en", "la", "las", "los", "o", "para", "por", "y"}
    expected_terms = set(_tokenize(expected_topic)) - stopwords
    return expected_terms.issubset(_tokenize(text))


def _build_corpus() -> tuple[list[str], list[dict[str, str]]]:
    """Build a small, known-good insurance corpus with one expected answer per question for the evaluation."""
    vocabulary = _build_vocabulary()
    corpus = [
        {
            "doc_id": "doc-1",
            "chunk_id": "chunk-1",
            "text": "La póliza cubre daños materiales del vehículo por accidente, robo o colisión.",
            "expected_topic": "danos materiales",
        },
        {
            "doc_id": "doc-1",
            "chunk_id": "chunk-2",
            "text": "La asistencia en carretera incluye remolque, cambio de neumático y ayuda mecánica.",
            "expected_topic": "asistencia en carretera",
        },
        {
            "doc_id": "doc-2",
            "chunk_id": "chunk-3",
            "text": "La cobertura médica incluye gastos hospitalarios, farmacia y reembolso por urgencias.",
            "expected_topic": "gastos hospitalarios",
        },
        {
            "doc_id": "doc-2",
            "chunk_id": "chunk-4",
            "text": "La cláusula de exclusiones elimina daños por uso indebido, negligencia o actos fraudulentos.",
            "expected_topic": "exclusiones",
        },
        {
            "doc_id": "doc-3",
            "chunk_id": "chunk-5",
            "text": "La vigencia de la póliza es anual y el deducible aplica solo a daños materiales.",
            "expected_topic": "vigencia y deducible",
        },
    ]
    return vocabulary, corpus


def _evaluate_single_question(question: str, expected_topic: str, k: int = 3) -> dict[str, object]:
    """Rank the small corpus for one question and report whether the expected answer appears in the top-k hits."""
    vocabulary, corpus = _build_corpus()
    query_vector = _embedding_from_text(question, vocabulary)
    ranked = sorted(
        corpus,
        key=lambda item: _cosine_similarity(query_vector, _embedding_from_text(item["text"], vocabulary)),
        reverse=True,
    )

    top_hits = ranked[:k]
    hit = any(_matches_expected_topic(expected_topic, item["text"]) for item in top_hits)
    return {
        "question": question,
        "expected_topic": expected_topic,
        "top_k": [item["chunk_id"] for item in top_hits],
        "hit": hit,
    }


def evaluate_retrieval() -> dict[str, object]:
    """Run the mini retrieval benchmark and summarize hit rate@3 over a small set of representative questions."""
    questions = [
        {"question": "¿Qué cubre la póliza en caso de daños materiales del vehículo?", "expected_topic": "danos materiales"},
        {"question": "¿Qué pasa si necesito asistencia en carretera?", "expected_topic": "asistencia en carretera"},
        {"question": "¿Incluye gastos médicos y reembolso por urgencias?", "expected_topic": "gastos hospitalarios"},
        {"question": "¿Qué exclusiones aparecen en la póliza?", "expected_topic": "exclusiones"},
        {"question": "¿Durante cuánto tiempo dura la cobertura y cómo funciona el deducible?", "expected_topic": "vigencia y deducible"},
    ]

    results = [_evaluate_single_question(item["question"], item["expected_topic"], k=3) for item in questions]
    hits = sum(1 for item in results if item["hit"])
    summary = {
        "k": 3,
        "n_questions": len(questions),
        "hits": hits,
        "hit_rate_at_k": round(hits / len(questions), 4),
        "results": results,
    }
    return summary


def main() -> int:
    """Print the small retrieval benchmark and fail fast if the synthetic sanity threshold is not met."""
    report = evaluate_retrieval()
    print(json.dumps(report, ensure_ascii=False, indent=2))

    threshold = 0.8
    if report["hit_rate_at_k"] < threshold:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
