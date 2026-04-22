from mappings import Mappings


def test_anonimizer() -> None:
    text = "my name is maxime, i live in France and my email address is maximebatista18@gmail.com"

    mappings = Mappings()
    text = anonymizer.anonymize_text(text, mappings)

    print(mappings._redacted_to_sensitive)
    print(mappings._sensitive_to_redacted)

    print("anonymized text", text)

    text = anonymizer.deanonymize_text(text, mappings)

    print("deanonymized text", text)
