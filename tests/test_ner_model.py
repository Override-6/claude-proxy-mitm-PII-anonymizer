"""
Edge-case tests for the fine-tuned bert-multilingual-uncased-ner model.

Covers English and French sentences including tricky cases such as titles,
possessives, compound names, punctuation adjacency, and mixed-language text.
The model is loaded once per session; each test asserts that a set of
expected (span, label) pairs are detected.
"""

from __future__ import annotations

import pathlib
import pytest
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# ---------------------------------------------------------------------------
# Model path
# ---------------------------------------------------------------------------

_MODEL_DIR = str(
    pathlib.Path(__file__).parent.parent / "models" / "bert-multilingual-uncased-ner"
)

# WikiANN label names after aggregation_strategy="simple"
# B-PER / I-PER → entity_group "PER", etc.
PER = "PER"
ORG = "ORG"
LOC = "LOC"


# ---------------------------------------------------------------------------
# Shared fixture – load once for the whole session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ner_pipe():
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(_MODEL_DIR)
    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=-1,  # CPU; change to 0 for GPU
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _find(results: list[dict], text: str, label: str) -> bool:
    """Return True if *text* (case-insensitive substring) with *label* is present."""
    needle = text.lower()
    for r in results:
        if r["entity_group"] == label and needle in r["word"].lower():
            return True
    return False


def _entities(ner_pipe, sentence: str) -> list[dict]:
    return ner_pipe(sentence)


# ---------------------------------------------------------------------------
# English – basic
# ---------------------------------------------------------------------------


class TestEnglishBasic:
    def test_person_simple(self, ner_pipe):
        results = _entities(ner_pipe, "Barack Obama was born in Hawaii.")
        assert _find(results, "Obama", PER), f"Expected PER 'Obama' in {results}"
        assert _find(results, "Hawaii", LOC), f"Expected LOC 'Hawaii' in {results}"

    def test_org_and_founders(self, ner_pipe):
        results = _entities(
            ner_pipe,
            "Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, California.",
        )
        assert _find(results, "Apple", ORG), f"Expected ORG 'Apple' in {results}"
        assert _find(results, "Steve Jobs", PER), f"Expected PER 'Steve Jobs' in {results}"
        assert _find(results, "Wozniak", PER), f"Expected PER 'Wozniak' in {results}"
        assert _find(results, "California", LOC), f"Expected LOC 'California' in {results}"

    def test_location_capital(self, ner_pipe):
        results = _entities(ner_pipe, "Paris is the capital of France.")
        assert _find(results, "Paris", LOC), f"Expected LOC 'Paris' in {results}"
        assert _find(results, "France", LOC), f"Expected LOC 'France' in {results}"

    def test_multiword_location(self, ner_pipe):
        results = _entities(ner_pipe, "New York City is located in the United States.")
        assert _find(results, "New York", LOC), f"Expected LOC 'New York' in {results}"
        assert _find(results, "United States", LOC), f"Expected LOC 'United States' in {results}"


# ---------------------------------------------------------------------------
# English – edge cases
# ---------------------------------------------------------------------------


class TestEnglishEdgeCases:
    def test_person_with_title(self, ner_pipe):
        """Title 'Dr.' should not prevent detection of the person."""
        results = _entities(ner_pipe, "Dr. Angela Merkel visited Berlin last Tuesday.")
        assert _find(results, "Merkel", PER), f"Expected PER 'Merkel' in {results}"
        assert _find(results, "Berlin", LOC), f"Expected LOC 'Berlin' in {results}"

    def test_possessive_org(self, ner_pipe):
        """'Microsoft's' — the possessive form should still yield an ORG."""
        results = _entities(ner_pipe, "Microsoft's CEO Satya Nadella announced the acquisition.")
        assert _find(results, "Microsoft", ORG), f"Expected ORG 'Microsoft' in {results}"
        assert _find(results, "Nadella", PER), f"Expected PER 'Nadella' in {results}"

    def test_entity_at_sentence_start(self, ner_pipe):
        results = _entities(ner_pipe, "Google announced major layoffs this quarter.")
        assert _find(results, "Google", ORG), f"Expected ORG 'Google' in {results}"

    def test_entity_at_sentence_end(self, ner_pipe):
        results = _entities(ner_pipe, "The conference will be held in London.")
        assert _find(results, "London", LOC), f"Expected LOC 'London' in {results}"

    def test_two_persons_same_sentence(self, ner_pipe):
        results = _entities(ner_pipe, "Tim Cook and Elon Musk met in San Francisco to discuss AI.")
        assert _find(results, "Tim Cook", PER), f"Expected PER 'Tim Cook' in {results}"
        assert _find(results, "Elon Musk", PER), f"Expected PER 'Elon Musk' in {results}"
        assert _find(results, "San Francisco", LOC), f"Expected LOC 'San Francisco' in {results}"

    def test_single_word_person(self, ner_pipe):
        results = _entities(ner_pipe, "Einstein proposed the theory of relativity.")
        assert _find(results, "Einstein", PER), f"Expected PER 'Einstein' in {results}"

    def test_single_word_location(self, ner_pipe):
        results = _entities(ner_pipe, "London is an amazing city to visit.")
        assert _find(results, "London", LOC), f"Expected LOC 'London' in {results}"

    def test_no_entities(self, ner_pipe):
        results = _entities(ner_pipe, "The weather is nice today and I feel great.")
        ner_hits = [r for r in results if r["entity_group"] in (PER, ORG, LOC)]
        assert len(ner_hits) == 0, f"Expected no NER entities, got {ner_hits}"

    def test_quoted_context(self, ner_pipe):
        results = _entities(
            ner_pipe, '"We need to leave Berlin immediately," said Angela Merkel.'
        )
        assert _find(results, "Berlin", LOC), f"Expected LOC 'Berlin' in {results}"
        assert _find(results, "Merkel", PER), f"Expected PER 'Merkel' in {results}"

    def test_person_with_jr(self, ner_pipe):
        results = _entities(ner_pipe, "Martin Luther King Jr. delivered the speech in Washington.")
        assert _find(results, "Martin Luther King", PER), f"Expected PER in {results}"
        assert _find(results, "Washington", LOC), f"Expected LOC 'Washington' in {results}"

    def test_person_in_parentheses(self, ner_pipe):
        results = _entities(ner_pipe, "The award was given to the director (Steven Spielberg).")
        assert _find(results, "Spielberg", PER), f"Expected PER 'Spielberg' in {results}"

    def test_hyphenated_location(self, ner_pipe):
        results = _entities(ner_pipe, "She grew up in Aix-en-Provence before moving to Paris.")
        assert _find(results, "Paris", LOC), f"Expected LOC 'Paris' in {results}"

    def test_acronym_org(self, ner_pipe):
        results = _entities(ner_pipe, "The CEO reported to the board of UNESCO.")
        assert _find(results, "UNESCO", ORG), f"Expected ORG 'UNESCO' in {results}"

    def test_person_comma_separated(self, ner_pipe):
        """WikiANN-style: comma after last I-PER token."""
        results = _entities(ner_pipe, "George Randolph Hearst, Jr. chaired the meeting.")
        assert _find(results, "Hearst", PER), f"Expected PER 'Hearst' in {results}"

    def test_org_vs_loc_amazon(self, ner_pipe):
        """'Amazon' in a corporate context should be ORG, not LOC."""
        results = _entities(ner_pipe, "Amazon reported record profits this year.")
        assert _find(results, "Amazon", ORG), f"Expected ORG 'Amazon' in {results}"

    def test_already_redacted_placeholder_ignored(self, ner_pipe):
        """Placeholders like [PERSON_0] must not be double-tagged as entities."""
        results = _entities(ner_pipe, "The email was sent by <SECRET_NOT_FOUND-[PERSON_0]> from [LOC_1].")
        # The placeholder text itself should not produce a spurious long entity
        for r in results:
            assert "PERSON_0" not in r["word"] and "LOC_1" not in r["word"], (
                f"Placeholder leaked into entity: {r}"
            )


# ---------------------------------------------------------------------------
# French – basic
# ---------------------------------------------------------------------------


class TestFrenchBasic:
    def test_person_and_location(self, ner_pipe):
        results = _entities(
            ner_pipe,
            "Emmanuel Macron est le président de la République française.",
        )
        assert _find(results, "Macron", PER), f"Expected PER 'Macron' in {results}"

    def test_org_french(self, ner_pipe):
        results = _entities(ner_pipe, "Air France a annoncé de nouveaux vols vers Tokyo.")
        assert _find(results, "Air France", ORG), f"Expected ORG 'Air France' in {results}"
        assert _find(results, "Tokyo", LOC), f"Expected LOC 'Tokyo' in {results}"

    def test_location_french_city(self, ner_pipe):
        results = _entities(ner_pipe, "Lyon est une grande ville située en France.")
        assert _find(results, "Lyon", LOC), f"Expected LOC 'Lyon' in {results}"
        assert _find(results, "France", LOC), f"Expected LOC 'France' in {results}"

    def test_two_persons_french(self, ner_pipe):
        results = _entities(
            ner_pipe,
            "Emmanuel Macron a rencontré Angela Merkel à Paris pour discuter de l'Europe.",
        )
        assert _find(results, "Macron", PER), f"Expected PER 'Macron' in {results}"
        assert _find(results, "Merkel", PER), f"Expected PER 'Merkel' in {results}"
        assert _find(results, "Paris", LOC), f"Expected LOC 'Paris' in {results}"

    def test_scientist_french(self, ner_pipe):
        results = _entities(
            ner_pipe, "Marie Curie est née à Varsovie et a travaillé à Paris."
        )
        assert _find(results, "Marie Curie", PER), f"Expected PER 'Marie Curie' in {results}"
        assert _find(results, "Varsovie", LOC), f"Expected LOC 'Varsovie' in {results}"
        assert _find(results, "Paris", LOC), f"Expected LOC 'Paris' in {results}"


# ---------------------------------------------------------------------------
# French – edge cases
# ---------------------------------------------------------------------------


class TestFrenchEdgeCases:
    def test_person_with_title_french(self, ner_pipe):
        results = _entities(
            ner_pipe, "Le président Nicolas Sarkozy a signé le traité à Lisbonne."
        )
        assert _find(results, "Sarkozy", PER), f"Expected PER 'Sarkozy' in {results}"
        assert _find(results, "Lisbonne", LOC), f"Expected LOC 'Lisbonne' in {results}"

    def test_org_grande_ecole(self, ner_pipe):
        results = _entities(ner_pipe, "Sciences Po est une grande école parisienne réputée.")
        assert _find(results, "Sciences Po", ORG), f"Expected ORG 'Sciences Po' in {results}"

    def test_location_compound_french(self, ner_pipe):
        results = _entities(
            ner_pipe, "Le Salon de Genève présente les derniers modèles automobiles."
        )
        assert _find(results, "Genève", LOC), f"Expected LOC 'Genève' in {results}"

    def test_no_entities_french(self, ner_pipe):
        results = _entities(ner_pipe, "Il fait beau aujourd'hui et je me sens bien.")
        ner_hits = [r for r in results if r["entity_group"] in (PER, ORG, LOC)]
        assert len(ner_hits) == 0, f"Expected no NER entities, got {ner_hits}"

    def test_person_end_of_sentence_french(self, ner_pipe):
        results = _entities(ner_pipe, "Le prix Nobel a été décerné à Albert Camus.")
        assert _find(results, "Camus", PER), f"Expected PER 'Camus' in {results}"

    def test_org_possessive_french(self, ner_pipe):
        results = _entities(
            ner_pipe, "Le PDG de Renault a démissionné suite au scandale."
        )
        assert _find(results, "Renault", ORG), f"Expected ORG 'Renault' in {results}"


# ---------------------------------------------------------------------------
# Mixed language
# ---------------------------------------------------------------------------


class TestMixedLanguage:
    def test_english_person_french_context(self, ner_pipe):
        results = _entities(ner_pipe, "Elon Musk dirige Tesla depuis San Francisco.")
        assert _find(results, "Elon Musk", PER), f"Expected PER 'Elon Musk' in {results}"
        assert _find(results, "Tesla", ORG), f"Expected ORG 'Tesla' in {results}"
        assert _find(results, "San Francisco", LOC), f"Expected LOC 'San Francisco' in {results}"

    def test_multilingual_persons(self, ner_pipe):
        results = _entities(
            ner_pipe,
            "Barack Obama et François Hollande se sont rencontrés à Washington.",
        )
        assert _find(results, "Obama", PER), f"Expected PER 'Obama' in {results}"
        assert _find(results, "Hollande", PER), f"Expected PER 'Hollande' in {results}"
        assert _find(results, "Washington", LOC), f"Expected LOC 'Washington' in {results}"
