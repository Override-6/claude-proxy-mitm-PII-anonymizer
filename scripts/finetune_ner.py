#!/usr/bin/env python3
"""
Fine-tune FacebookAI/xlm-roberta-base for NER (PER, ORG, LOC).

All tokens are lowercased before tokenization — simulates an uncased model
without needing a dedicated uncased checkpoint.

Datasets (positive examples):
  - wikiann                en + fr   WikiANN, PER/ORG/LOC
  - conll2003              en        CoNLL-2003, MISC dropped
  - Jean-Baptiste/wikiner_fr         French WikiNER, MISC dropped
  - Babelscape/multinerd   en + fr   Fine-grained multilingual NER (40+ types → PER/ORG/LOC)
  - tner/ontonotes5        en        Diverse domains: web/broadcast/phone/news

Hard negatives (all O — teach the model what is NOT an entity):
  - codeparrot/github-code Python     Source code with string literals stripped
  - data/requests-sample.jsonl        Real proxy traffic (tool descriptions, code, markdown)
  - Synthetic                          Generated technical text patterns

Labels: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC
"""

import argparse
import itertools
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import datasets as _datasets_mod

_datasets_mod.disable_caching()  # Prevent stale cached intermediate remaps from poisoning the pipeline.
from datasets import ClassLabel, Dataset, Sequence, concatenate_datasets, load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Label scheme
# ---------------------------------------------------------------------------

LABEL_LIST = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}

MODEL_NAME = "FacebookAI/xlm-roberta-base"

_TARGET_NER_FEATURE = Sequence(ClassLabel(names=LABEL_LIST))

# OntoNotes5 uses PERSON/GPE instead of PER/LOC — explicit remap needed.
_ONTONOTES_REMAP = {
    "B-PERSON": "B-PER", "I-PERSON": "I-PER",
    "B-GPE": "B-LOC", "I-GPE": "I-LOC",  # geo-political entity → LOC
    "B-ORG": "B-ORG", "I-ORG": "I-ORG",
    "B-LOC": "B-LOC", "I-LOC": "I-LOC",
    "O": "O",
    # everything else (DATE, CARDINAL, MONEY, …) → O implicitly via .get default
}


# ---------------------------------------------------------------------------
# Dataset normalisation helpers
# ---------------------------------------------------------------------------


def _normalize_dataset(ds, name_remap: dict | None = None):
    """
    Remap a dataset's ner_tags to our 7-label ID scheme and cast to
    _TARGET_NER_FEATURE so concatenate_datasets sees identical schemas.

    name_remap: optional {source_label_name: our_label_name} override,
                used for datasets whose label names differ (e.g. OntoNotes5).

    Handles BIO and IO schemes automatically.
    """
    features = ds.features["ner_tags"]

    # Cast to plain int32 immediately — removes ClassLabel range validation so
    # we can remap OOB values freely without the HF framework raising errors.
    from datasets import Value as DSValue
    ds = ds.cast_column("ner_tags", Sequence(DSValue("int32")))

    if not (hasattr(features, "feature") and hasattr(features.feature, "int2str")):
        # No ClassLabel metadata (e.g. Parquet mirrors like lhoestq/conll2003 and
        # Babelscape/multinerd).  Their first 7 IDs happen to match our scheme
        # (0=O, 1=B-PER, …, 6=I-LOC); anything ≥ 7 (MISC, ANIM, …) maps to O.
        _n = len(LABEL_LIST)

        def _clamp(example):
            example["ner_tags"] = [t if t < _n else 0 for t in example["ner_tags"]]
            return example

        ds = ds.map(_clamp, desc="Clamping OOB labels")
        return ds.select_columns(["tokens", "ner_tags"])

    cl = features.feature
    names = [cl.int2str(i) for i in range(cl.num_classes)]

    if name_remap:
        names = [name_remap.get(n, n) for n in names]

    is_io = any(
        n != "O" and not n.startswith("B-") and not n.startswith("I-")
        for n in names
    )

    if is_io:
        int2type = {i: n for i, n in enumerate(names)}

        def io_to_bio(example):
            bio_ids, prev = [], "O"
            for raw in example["ner_tags"]:
                typ = int2type.get(int(raw), "O")
                if typ == "O":
                    bio_ids.append(LABEL2ID["O"])
                    prev = "O"
                else:
                    prefix = "B" if typ != prev else "I"
                    bio_ids.append(LABEL2ID.get(f"{prefix}-{typ}", LABEL2ID["O"]))
                    prev = typ
            example["ner_tags"] = bio_ids
            return example

        ds = ds.map(io_to_bio, desc="IO→BIO", load_from_cache_file=False)
    else:
        remap = {i: LABEL2ID.get(n, LABEL2ID["O"]) for i, n in enumerate(names)}

        def apply_remap(example):
            example["ner_tags"] = [remap.get(int(t), 0) for t in example["ner_tags"]]
            return example

        ds = ds.map(apply_remap, desc="Remapping labels", load_from_cache_file=False)

    # Return as Sequence(int32) — caller does the final ClassLabel cast after
    # concatenate_datasets so HF schema validation fires only once on clean data.
    return ds.select_columns(["tokens", "ner_tags"])


def _lowercase(example):
    example["tokens"] = [t.lower() for t in example["tokens"]]
    return example


def _all_o(tokens: list[str]) -> dict:
    return {"tokens": tokens, "ner_tags": [LABEL2ID["O"]] * len(tokens)}


# ---------------------------------------------------------------------------
# Positive-example loaders
# ---------------------------------------------------------------------------


def _load_wikiann(split: str):
    parts = []
    for lang in ("en", "fr"):
        ds = load_dataset("wikiann", lang, split=split)
        parts.append(_normalize_dataset(ds))
    return concatenate_datasets(parts)


def _load_conll2003(split: str):
    # lhoestq/conll2003 is a script-free Parquet mirror of CoNLL-2003.
    hf_split = "validation" if split == "validation" else split
    ds = load_dataset("lhoestq/conll2003", split=hf_split)
    return _normalize_dataset(ds)


def _load_wikiner_fr(split: str):
    # wikiner_fr has only train/test — use test as validation.
    # Some examples have label IDs outside the ClassLabel range; cast to plain
    # int32 first so the HF framework doesn't raise a validation error.
    hf_split = "test" if split == "validation" else split
    try:
        from datasets import Sequence as DSSeq, Value
        ds = load_dataset("Jean-Baptiste/wikiner_fr", split=hf_split)
        ds = ds.cast_column("ner_tags", DSSeq(Value("int32")))
        # Rebuild a name→id mapping from the original features before the cast.
        # wikiner_fr IO labels: 0=O, 1=PER, 2=ORG, 3=LOC, 4=MISC
        int2type = {0: "O", 1: "PER", 2: "ORG", 3: "LOC", 4: "MISC"}

        def io_to_bio(example):
            bio_ids, prev = [], "O"
            for raw in example["ner_tags"]:
                typ = int2type.get(int(raw), "O")
                if typ in ("O", "MISC"):
                    bio_ids.append(LABEL2ID["O"])
                    prev = "O"
                else:
                    prefix = "B" if typ != prev else "I"
                    bio_ids.append(LABEL2ID.get(f"{prefix}-{typ}", LABEL2ID["O"]))
                    prev = typ
            example["ner_tags"] = bio_ids
            return example

        ds = ds.map(io_to_bio, desc="wikiner_fr IO→BIO")
        return ds.select_columns(["tokens", "ner_tags"])
    except Exception as e:
        print(f"  [warn] wikiner_fr not available ({e}), skipping.")
        return None


_MULTINERD_CAP = 200_000  # multinerd has 1.3M examples; cap to keep dataset balance


def _load_multinerd(split: str):
    """
    Babelscape/multinerd — fine-grained multilingual NER with 16 entity types.
    We keep PER/ORG/LOC and drop the rest (ANIM, BIO, CEL, DIS, EVE, …) to O.
    Includes English and French (and 8 other languages — diversity helps).
    Capped at _MULTINERD_CAP to prevent it from dominating the training mix.
    """
    try:
        hf_split = "validation" if split == "validation" else split
        ds = load_dataset("Babelscape/multinerd", split=hf_split,
                          verification_mode="no_checks")
        cap = _MULTINERD_CAP if split == "train" else _MULTINERD_CAP // 5
        if len(ds) > cap:
            ds = ds.shuffle(seed=42).select(range(cap))
            print(f"    (capped multinerd to {cap:,} examples)")
        return _normalize_dataset(ds)
    except Exception as e:
        print(f"  [warn] multinerd not available ({e}), skipping.")
        return None


def _load_ontonotes5(split: str):
    """
    tner/ontonotes5 — English NER across broadcast, web, phone, news, magazine.
    Much more domain-diverse than WikiANN.  Uses PERSON/GPE instead of PER/LOC.
    """
    try:
        # DFKI-SLT/few-nerd — large broad-domain NER with 66 fine-grained types.
        # Our normaliser maps unknown types to O, keeping only PER/ORG/LOC.
        hf_split = "validation" if split == "validation" else split
        ds = load_dataset("DFKI-SLT/few-nerd", "supervised", split=hf_split)
        return _normalize_dataset(ds)
    except Exception as e:
        print(f"  [warn] few-nerd not available ({e}), skipping.")
        return None


# ---------------------------------------------------------------------------
# Hard-negative loaders (all O)
# ---------------------------------------------------------------------------

# Strip string literals AND comments — both can contain real prose with names.
# Order matters: strip multi-line strings before single-line, then comments.
_STRIP_RE = re.compile(
    r'"""[\s\S]*?"""|'  # triple-double-quoted strings
    r"'''[\s\S]*?'''|"  # triple-single-quoted strings
    r'"[^"\\\n]*(?:\\.[^"\\\n]*)*"|'  # double-quoted strings
    r"'[^'\\\n]*(?:\\.[^'\\\n]*)*'|"  # single-quoted strings
    r"#[^\n]*|"  # single-line comments (#...)
    r"//[^\n]*|"  # single-line comments (//...)
    r"/\*[\s\S\n]*\*/",  # multi-line comments (//...)
    re.DOTALL,
)
# Extract identifier-like tokens (no digits-only, no single chars)
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")

_CODE_LANGUAGES = ["python", "java", "javascript", "go", "ruby", "php"]


def _load_code_negatives(n: int) -> Dataset | None:
    """
    Load source code from code-search-net (Python/Java/JS/Go/Ruby/PHP), strip
    all string literals and comments, then use the pre-tokenized code tokens
    directly and label everything O.

    code_search_net provides func_code_tokens (tokenized function body without
    docstring) — ideal since it already excludes natural-language documentation.
    Rust, Haskell, and TypeScript are not in this dataset; their idiomatic
    identifiers are covered by _make_synthetic_negatives instead.

    This teaches the model that package names, function names, class names,
    snake_case / camelCase / PascalCase identifiers are not entities.
    """
    per_lang = max(1, n // len(_CODE_LANGUAGES))
    all_records = []

    for lang in _CODE_LANGUAGES:
        try:
            ds_iter = load_dataset(
                "code-search-net/code_search_net",
                lang,
                split="train",
                streaming=True,
            )
        except Exception as e:
            print(f"  [warn] code_search_net/{lang} not available ({e}), skipping.")
            continue

        lang_records = []
        for example in itertools.islice(ds_iter, per_lang * 5):
            # func_code_tokens: pre-tokenized code without docstring — no NL prose.
            tokens = example.get("func_code_tokens", [])
            # Keep only identifier-like tokens; skip punctuation, numbers, keywords.
            idents = [t.lower() for t in tokens if _IDENT_RE.fullmatch(t) and len(t) > 1]
            if 6 <= len(idents) <= 40:
                lang_records.append(_all_o(idents[:40]))
            if len(lang_records) >= per_lang:
                break

        all_records.extend(lang_records)

    if not all_records:
        return None
    return Dataset.from_list(all_records)


def _load_request_negatives(jsonl_path: Path, max_examples: int = 2000) -> Dataset | None:
    """
    Extract *system-prompt / tool-schema* chunks from requests-sample.jsonl
    and label them all O.

    Only fields that are structurally guaranteed to be non-entity text are used:
      - system prompt (injected by the proxy itself, no user PII)
      - tool descriptions and parameter schemas

    We deliberately skip message content (user/assistant turns) because those
    may contain real names and addresses — labelling them O would actively teach
    the model to ignore legitimate entities.
    """
    if not jsonl_path.exists():
        print(f"  [warn] {jsonl_path} not found, skipping request negatives.")
        return None

    _TOKEN_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_\-\.]*[A-Za-z0-9])?")

    def _safe_chunks(text: str) -> list[list[str]]:
        tokens = _TOKEN_RE.findall(text.lower())
        return [tokens[i:i + 15] for i in range(0, len(tokens) - 6, 10)
                if len(tokens[i:i + 15]) >= 6]

    def _walk_schema(obj) -> list[str]:
        """Pull description strings out of a JSON Schema object recursively."""
        texts = []
        if isinstance(obj, dict):
            if "description" in obj and isinstance(obj["description"], str):
                texts.append(obj["description"])
            for v in obj.values():
                texts.extend(_walk_schema(v))
        elif isinstance(obj, list):
            for item in obj:
                texts.extend(_walk_schema(item))
        return texts

    records = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            body = entry.get("body", "")
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except Exception:
                    continue
            if not isinstance(body, dict):
                continue

            # System prompt field — injected by the proxy, no user PII.
            system = body.get("system", "")
            if isinstance(system, str) and len(system) > 30:
                for chunk in _safe_chunks(system):
                    records.append(_all_o(chunk))

            # Tool schemas — parameter descriptions contain no entity names.
            for tool in body.get("tools", []):
                for text in _walk_schema(tool.get("input_schema", {})):
                    if len(text) > 15:
                        for chunk in _safe_chunks(text):
                            records.append(_all_o(chunk))

            if len(records) >= max_examples:
                break

    if not records:
        return None
    rng = random.Random(0)
    rng.shuffle(records)
    return Dataset.from_list(records[:max_examples])


def _make_synthetic_negatives(n: int, rng: random.Random) -> Dataset:
    """
    Synthetic all-O examples: tool names, package names, identifiers, UUIDs,
    markdown fragments, and French/English connective words.
    """
    _TECH = [
        "taskcreate", "taskupdate", "webfetch", "websearch", "crondelete",
        "enterplanmode", "exitplanmode", "notebookedit", "remotetrigger",
        "autotokenizer", "trainerarguments", "datacollatortokenclassification",
        "xlm-roberta-base", "bert-base-multilingual-uncased",
        "mitmproxy", "mitmdump", "easyocr", "presidio-analyzer", "flashtext",
        "pyahocorasick", "transformers", "accelerate", "seqeval", "datasets",
        "aggregation_strategy", "is_split_into_words", "word_ids",
        "label2id", "id2label", "num_labels", "entity_group",
        "b-per", "i-per", "b-org", "i-org", "b-loc", "i-loc",
        "docker-compose.yml", "pyproject.toml", "settings.json", "poetry.lock",
        ">=11.0.0", ">=1.7.0", "^3.12", "^5.4.0", "^2.3.0",
        "0.0.0.0:8080", "127.0.0.1:9999", "localhost:8000",
        "sha256:a1b2c3d4", "c7f1b075-4e95-4bdd-9fa2-4921d937d03c",
        "##", "###", "```python", "```bash", "|---|", "- [ ]",
        "string", "boolean", "integer", "object", "array", "null",
        "rust", "java", "kotlin", "swift", "scala", "golang",
        "shell", "bash", "grep", "curl", "npm", "pip", "cargo",
        "figma", "figjam", "gmail", "claude", "anthropic",
        "misc", "prod", "dev", "stag", "loc", "per", "eve",
        # Rust idioms
        "fn", "impl", "struct", "enum", "trait", "pub", "mut", "let",
        "vec", "hashmap", "result", "option", "unwrap", "clone", "iter",
        "tokio", "serde", "anyhow", "thiserror", "reqwest", "actix",
        # Haskell idioms
        "data", "where", "deriving", "instance", "class", "module",
        "maybe", "either", "monad", "functor", "applicative", "foldable",
        "ghc", "cabal", "stack", "parsec", "aeson", "conduit",
        # TypeScript idioms
        "interface", "readonly", "keyof", "typeof", "infer", "extends",
        "generic", "partial", "required", "record", "promise", "async",
        "react", "nextjs", "prisma", "zod", "trpc", "tailwind",
        "nat", "ide", "api", "sdk", "cli", "url", "uri", "sse",
        "update-config", "keybindings-help", "simplify",
        "src/proxy.py", "/app/models", "/root/.cache/huggingface",
    ]
    _CONNECTIVES_EN = [
        "use", "the", "to", "in", "of", "a", "is", "are", "was", "has",
        "with", "for", "this", "that", "run", "call", "returns", "loads",
        "see", "check", "note", "set", "get", "add", "remove", "update",
        "when", "if", "then", "and", "or", "not", "via", "as", "from",
        "available", "using", "based", "only", "first", "new", "each",
    ]
    _CONNECTIVES_FR = [
        "utilise", "le", "la", "les", "de", "du", "des", "un", "une",
        "pour", "dans", "avec", "sur", "par", "est", "sont", "peut",
        "cette", "ici", "aussi", "alors", "mais", "comme", "bien",
    ]
    connectives = _CONNECTIVES_EN + _CONNECTIVES_FR

    records = []
    for _ in range(n):
        length = rng.randint(6, 20)
        tokens = [
            rng.choice(_TECH).lower() if rng.random() < 0.5
            else rng.choice(connectives)
            for _ in range(length)
        ]
        records.append(_all_o(tokens))

    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Synthetic positive examples
# ---------------------------------------------------------------------------


def _make_synthetic_positives(n: int, rng: random.Random) -> Dataset:
    """
    Synthetic NER examples that teach the model to detect entities in
    technical / proxy-traffic contexts: JSON key-value pairs, tool output,
    markdown, French addresses.

    All tokens are lowercased to match the inference pipeline.
    """
    # ---- Word lists --------------------------------------------------------
    FIRST_FR = ["jean", "pierre", "marie", "sophie", "thomas", "alice", "nicolas",
                "isabelle", "francois", "claire", "olivier", "anne", "paul", "julie",
                "michel", "laurie", "henri", "margot", "luc", "emilie", "maxime",
                "camille", "antoine", "lucie", "romain", "elodie", "fabien", "aurelie"]
    FIRST_EN = ["john", "james", "mary", "sarah", "david", "emma", "michael", "lisa",
                "william", "anna", "robert", "emily", "charles", "kate", "george",
                "helen", "peter", "diana", "mark", "laura", "kevin", "rachel", "brian"]
    LAST_FR = ["martin", "bernard", "thomas", "petit", "robert", "richard", "durand",
               "dubois", "moreau", "laurent", "simon", "michel", "lefebvre", "leroy",
               "roux", "david", "bertrand", "morel", "fournier", "girard", "bonnet"]
    LAST_EN = ["smith", "jones", "williams", "brown", "taylor", "davies", "wilson",
               "evans", "johnson", "miller", "anderson", "clark", "thompson", "white",
               "harris", "hall", "walker", "young", "allen", "king", "wright", "scott"]

    ORGS = [
        ["societe", "generale"],
        ["credit", "agricole"],
        ["bnp", "paribas"],
        ["medecins", "sans", "frontieres"],
        ["croix", "rouge"],
        ["ecole", "polytechnique"],
        ["universite", "paris", "saclay"],
        ["mairie", "de", "paris"],
        ["chambre", "de", "commerce"],
        ["agence", "nationale"],
        ["hopital", "lariboisiere"],
        ["caisse", "depargne"],
        ["la", "poste"],
        ["france", "televisions"],
        ["acme", "corporation"],
        ["globex", "industries"],
        ["initech", "solutions"],
        ["veridian", "dynamics"],
    ]

    STREET_TYPES = ["rue", "avenue", "boulevard", "place", "impasse", "chemin", "allee",
                    "passage", "cite", "square", "voie", "route"]
    STREET_NAMES = ["de", "la", "republique", "du", "general", "de", "gaulle",
                    "jean", "jaures", "victor", "hugo", "pasteur", "liberte",
                    "du", "commerce", "de", "la", "paix", "nationale",
                    "du", "marechal", "foch", "des", "ecoles", "du", "moulin"]
    CITIES_FR = ["paris", "lyon", "marseille", "bordeaux", "toulouse", "nantes",
                 "strasbourg", "lille", "rennes", "montpellier", "nice", "grenoble",
                 "dijon", "metz", "brest", "rouen", "caen", "limoges"]
    CITIES_EN = ["london", "berlin", "madrid", "rome", "brussels", "amsterdam",
                 "zurich", "geneva", "montreal", "new", "york", "boston", "chicago"]

    # ---- Context templates -------------------------------------------------
    # Each template is a function that returns (tokens: list[str], labels: list[int])
    # where labels are integer IDs from LABEL2ID.

    O = LABEL2ID["O"]
    B_PER = LABEL2ID["B-PER"];
    I_PER = LABEL2ID["I-PER"]
    B_ORG = LABEL2ID["B-ORG"];
    I_ORG = LABEL2ID["I-ORG"]
    B_LOC = LABEL2ID["B-LOC"];
    I_LOC = LABEL2ID["I-LOC"]

    def _per(first, last):
        return [first, last], [B_PER, I_PER]

    def _org(parts):
        return parts, [B_ORG] + [I_ORG] * (len(parts) - 1)

    def _loc(parts):
        return parts, [B_LOC] + [I_LOC] * (len(parts) - 1)

    def _city():
        c = rng.choice(CITIES_FR + CITIES_EN)
        if isinstance(c, list):
            return c, [B_LOC] + [I_LOC] * (len(c) - 1)
        return [c], [B_LOC]

    def _address():
        """Generate a French street address as a LOC span."""
        num = str(rng.randint(1, 250))
        stype = rng.choice(STREET_TYPES)
        # 2-4 street name words
        sname = rng.sample(STREET_NAMES, rng.randint(2, 4))
        city = rng.choice(CITIES_FR)
        parts = [num, stype] + sname + [city]
        return parts, [B_LOC] + [I_LOC] * (len(parts) - 1)

    def _rand_person():
        pool_f = FIRST_FR if rng.random() < 0.6 else FIRST_EN
        pool_l = LAST_FR if rng.random() < 0.6 else LAST_EN
        return rng.choice(pool_f), rng.choice(pool_l)

    # Template builders — each returns a full (tokens, labels) pair
    def tpl_json_author():
        f, l = _rand_person()
        pt, pl = _per(f, l)
        pre = rng.choice(["author", "user", "owner", "contact", "assignee",
                          "created", "by", "modified", "by", "from"])
        return [pre] + pt, [O] + pl

    def tpl_json_org():
        org = rng.choice(ORGS)
        ot, ol = _org(org)
        pre = rng.choice(["company", "organization", "client", "partner",
                          "employer", "institution", "provider"])
        return [pre] + ot, [O] + ol

    def tpl_json_location():
        ct, cl = _city()
        pre = rng.choice(["city", "location", "country", "region",
                          "office", "based", "in", "from"])
        return [pre] + ct, [O] + cl

    def tpl_json_address():
        at, al = _address()
        pre = rng.choice(["address", "domicile", "livraison", "facturation",
                          "siege", "social", "at", "located", "at"])
        return [pre] + at, [O] + al

    def tpl_sentence_per():
        f, l = _rand_person()
        pt, pl = _per(f, l)
        pre = rng.choice([
            ["the", "document", "was", "authored", "by"],
            ["email", "received", "from"],
            ["assigned", "to"],
            ["contact"],
            ["shared", "with"],
            ["created", "by"],
            ["owned", "by"],
            ["le", "document", "a", "ete", "cree", "par"],
            ["envoye", "par"],
            ["contact"],
            ["attribue", "a"],
            ["gere", "par"],
        ])
        suf = rng.choice([[], ["for", "review"], ["regarding", "the", "request"],
                          ["via", "email"], ["concerning", "the", "project"]])
        return pre + pt + suf, [O] * len(pre) + pl + [O] * len(suf)

    def tpl_sentence_org():
        org = rng.choice(ORGS)
        ot, ol = _org(org)
        pre = rng.choice([
            ["the", "client"],
            ["our", "partner"],
            ["le", "client"],
            ["la", "societe"],
            ["notre", "partenaire"],
            ["invoice", "from"],
            ["facture", "de"],
            ["contrat", "avec"],
            ["contract", "with"],
        ])
        suf = rng.choice([[], ["has", "requested"], ["a", "confirme"],
                          ["responded"], ["signed", "the", "contract"]])
        return pre + ot + suf, [O] * len(pre) + ol + [O] * len(suf)

    def tpl_sentence_loc():
        if rng.random() < 0.5:
            ct, cl = _city()
            pre = rng.choice([
                ["the", "office", "is", "located", "in"],
                ["shipping", "to"],
                ["meeting", "in"],
                ["headquartered", "in"],
                ["le", "bureau", "est", "situe", "a"],
                ["livraison", "vers"],
                ["reunion", "a"],
            ])
        else:
            ct, cl = _address()
            pre = rng.choice([
                ["address"],
                ["domicile"],
                ["siege", "social"],
                ["livre", "a"],
                ["delivered", "to"],
                ["located", "at"],
            ])
        return pre + ct, [O] * len(pre) + cl

    def tpl_multi_entity():
        """Sentence with both PER and ORG or PER and LOC."""
        f, l = _rand_person()
        pt, pl = _per(f, l)
        if rng.random() < 0.5:
            org = rng.choice(ORGS)
            ot, ol = _org(org)
            mid = rng.choice([["works", "at"], ["employed", "by"],
                              ["est", "employe", "chez"], ["travaille", "pour"]])
            return pt + mid + ot, pl + [O] * len(mid) + ol
        else:
            ct, cl = _city()
            mid = rng.choice([["lives", "in"], ["based", "in"],
                              ["habite", "a"], ["reside", "a"]])
            return pt + mid + ct, pl + [O] * len(mid) + cl

    templates = [
        tpl_json_author, tpl_json_org, tpl_json_location, tpl_json_address,
        tpl_sentence_per, tpl_sentence_per, tpl_sentence_per,  # weight PER higher
        tpl_sentence_org, tpl_sentence_loc, tpl_sentence_loc,
        tpl_multi_entity, tpl_multi_entity,
    ]

    records = []
    for _ in range(n):
        tpl = rng.choice(templates)
        tokens, labels = tpl()
        records.append({"tokens": tokens, "ner_tags": labels})

    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Combined loader
# ---------------------------------------------------------------------------


def load_all(split: str) -> Dataset:
    parts = []

    # --- Positive examples ---
    print(f"  wikiann en+fr ({split}) …")
    parts.append(_load_wikiann(split))

    print(f"  conll2003 ({split}) …")
    try:
        parts.append(_load_conll2003(split))
    except Exception as e:
        print(f"  [warn] conll2003 failed ({e}), skipping.")

    print(f"  wikiner_fr ({split}) …")
    ds = _load_wikiner_fr(split)
    if ds is not None:
        parts.append(ds)

    print(f"  multinerd ({split}) …")
    ds = _load_multinerd(split)
    if ds is not None:
        parts.append(ds)

    print(f"  ontonotes5 ({split}) …")
    ds = _load_ontonotes5(split)
    if ds is not None:
        parts.append(ds)

    # --- Hard negatives ---
    if split == "train":
        print("  code negatives (github-code Python, ~5000) …")
        ds = _load_code_negatives(50000)
        if ds is not None:
            parts.append(ds)
    else:
        print("  code negatives (github-code Python, ~500) …")
        ds = _load_code_negatives(5000)
        if ds is not None:
            parts.append(ds)

    neg_n = 5000 if split == "train" else 500
    print(f"  synthetic negatives (n={neg_n}) …")
    parts.append(_make_synthetic_negatives(neg_n, random.Random(42)))

    # Normalise every part to Sequence(int32) before concatenation.
    # Dataset.from_list() defaults Python ints to int64; map() calls may
    # produce int64 too.  All must match before concatenate_datasets() can align
    # schemas, and we do the final ClassLabel cast on the merged dataset.
    from datasets import Value as DSValue
    _int32_seq = Sequence(DSValue("int32"))
    parts = [p.cast_column("ner_tags", _int32_seq) for p in parts]

    combined = concatenate_datasets(parts)
    # Single ClassLabel cast after all datasets are merged — avoids HF's
    # per-column range validation firing on stale intermediate ClassLabel state.
    combined = combined.cast_column("ner_tags", _TARGET_NER_FEATURE)
    combined = combined.map(_lowercase, desc="Lowercasing")
    return combined


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def tokenize_and_align(examples, tokenizer):
    """
    Tokenize pre-split word lists and realign BIO labels to sub-word tokens.

    First sub-word → original label.
    Continuation sub-words → I- equivalent (not -100).

    Using I- for continuations teaches the model span-boundary prediction,
    preventing subword fragmentation at inference (e.g. "charlie" split into
    ["char","lie"], both predicted B-PER, creating broken spans the
    word-boundary filter then discards entirely).
    """
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=256,
    )
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(labels[word_id])
            else:
                orig = LABEL_LIST[labels[word_id]]
                if orig.startswith("B-"):
                    label_ids.append(LABEL2ID["I-" + orig[2:]])
                elif orig.startswith("I-"):
                    label_ids.append(labels[word_id])
                else:
                    label_ids.append(-100)
            prev_word_id = word_id
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized


# ---------------------------------------------------------------------------
# Metrics / diagnostics
# ---------------------------------------------------------------------------


def build_compute_metrics():
    from seqeval.metrics import f1_score, precision_score, recall_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        true_labels, true_preds = [], []
        for pred_seq, label_seq in zip(predictions, labels):
            t_labels, t_preds = [], []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                t_labels.append(LABEL_LIST[l])
                t_preds.append(LABEL_LIST[p])
            true_labels.append(t_labels)
            true_preds.append(t_preds)
        return {
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
            "f1": f1_score(true_labels, true_preds),
        }

    return compute_metrics


def show_sample_predictions(model, tokenizer, dataset, n=5, device="cpu"):
    model.eval()
    model.to(device)
    print(f"\n--- Sample predictions (first {n} validation examples) ---")
    for idx in range(min(n, len(dataset))):
        tokens = dataset[idx]["tokens"]
        ner_tags = dataset[idx]["ner_tags"]
        encoded = tokenizer(
            tokens, is_split_into_words=True, return_tensors="pt",
            truncation=True, max_length=256,
        ).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
        pred_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()
        word_ids = encoded.word_ids(batch_index=0)
        pred_labels, prev_wid = [], None
        for wid, pid in zip(word_ids, pred_ids):
            if wid is None or wid == prev_wid:
                prev_wid = wid
                continue
            pred_labels.append(LABEL_LIST[pid])
            prev_wid = wid
        gold_labels = [LABEL_LIST[t] for t in ner_tags]
        print(f"\nExample {idx + 1}:")
        for tok, gold, pred in zip(tokens, gold_labels, pred_labels):
            marker = " " if gold == pred else "*"
            print(f"  {marker} {tok:20s}  gold={gold:6s}  pred={pred:6s}")


def print_label_distribution(dataset, tag: str):
    counts: Counter = Counter()
    for example in dataset:
        for label_id in example["ner_tags"]:
            counts[LABEL_LIST[label_id]] += 1
    total = sum(counts.values())
    print(f"\n--- Label distribution ({tag}, {len(dataset):,} examples) ---")
    if total > 0:
        for label in LABEL_LIST:
            c = counts.get(label, 0)
            print(f"  {label:6s}: {c:>9,}  ({100 * c / total:.1f}%)")
    print(f"  {'TOTAL':6s}: {total:>9,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune XLM-RoBERTa-base for multilingual NER (PER/ORG/LOC)"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--output_dir", type=str, default="./models/xlm-roberta-ner")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", type=bool, default=False)
    return parser.parse_args()


def main():
    args = parse_args()

    validate = args.validate

    if validate:
        MODEL_NAME = "models/xlm-roberta-ner"
    else:
        MODEL_NAME = "FacebookAI/xlm-roberta-base"

    print(f"Model        : {MODEL_NAME}")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch size   : {args.batch_size}")
    print(f"LR           : {args.learning_rate}")
    print(f"Output dir   : {args.output_dir}")
    print(f"Dry-run      : {args.dry_run}")
    print(f"Validate      : {args.validate}")

    print("\nLoading datasets …")
    train_raw = Dataset.from_dict({
        "tokens": [],
        "ner_tags": []
    })
    if not validate:
        train_raw = load_all("train")

    val_raw = load_all("validation")
    print(f"\nTrain: {len(train_raw):,}  |  Val: {len(val_raw):,}")
    print_label_distribution(train_raw, "train")

    print("\nLoading tokenizer and model …")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    print("Tokenizing …")
    train_ds = train_raw.map(
        lambda ex: tokenize_and_align(ex, tokenizer),
        batched=True, remove_columns=train_raw.column_names, desc="Tokenizing train",
    )
    val_ds = val_raw.map(
        lambda ex: tokenize_and_align(ex, tokenizer),
        batched=True, remove_columns=val_raw.column_names, desc="Tokenizing val",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    effective_epochs = 1 if args.dry_run else args.epochs

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=effective_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        eval_strategy="no" if args.dry_run else "epoch",
        save_strategy="no" if args.dry_run else "epoch",
        load_best_model_at_end=not args.dry_run,
        metric_for_best_model="f1",
        logging_steps=100,
        report_to="none",
        max_steps=1 if args.dry_run else -1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(),
    )

    if not validate:
        print("\nStarting training …")
        trainer.train()

    if args.dry_run:
        print("\n[dry-run] Done after 1 step.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        show_sample_predictions(model, tokenizer, val_raw, n=3, device=device)
        sys.exit(0)

    print("\nRunning evaluation …")
    metrics = trainer.evaluate()
    print(f"Eval metrics: {metrics}")

    print(f"\nSaving model to {args.output_dir} …")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    show_sample_predictions(model, tokenizer, val_raw, n=5, device=device)
    print("\nDone.")


if __name__ == "__main__":
    main()
