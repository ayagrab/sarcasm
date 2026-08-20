"""One place to make NLTK's tokenizer/sentiment resources available.

This machine's system CA bundle doesn't validate nltk.org's cert
(`CERTIFICATE_VERIFY_FAILED`) under the default `ssl` context, while
`certifi`'s bundle (already installed as a transitive dependency of
`requests`) works fine -- so we point `SSL_CERT_FILE` at it before the
first download. Once downloaded, resources are cached under
`~/nltk_data` and no further network access is needed.
"""
from __future__ import annotations

import os

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import nltk  # noqa: E402


def ensure_nltk_resources() -> None:
    for resource, package in [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("sentiment/vader_lexicon", "vader_lexicon"),
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)
