"""The evidence set: the bounded, labelled, de-duplicated passages handed to the LLM.

Each passage gets a short label (``E1``, ``E2`` ...). The model may only cite these labels, and
the validator checks every cited label against this set, so a citation can never point outside
the current evidence. Retrieved text is *untrusted data*: it is fenced with a random per-request
marker so that instructions hidden in an abstract or PDF cannot close the data block.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from biomentor.data.preprocessing import normalize_for_match
from biomentor.schemas import Passage, RetrievedPassage


@dataclass(frozen=True)
class EvidenceItem:
    label: str
    passage: Passage
    origin: str  # "retrieved" | "uploaded" | "external"
    score: float | None = None

    def describe(self) -> str:
        p = self.passage
        bits = [p.source_name or p.source_type]
        if p.pmid:
            bits.append(f"PMID {p.pmid}")
        if p.publication_year:
            bits.append(str(p.publication_year))
        if p.journal:
            bits.append(p.journal)
        return "; ".join(bits)


class EvidenceSet:
    def __init__(self, items: Sequence[EvidenceItem], nonce: str | None = None):
        self.items = list(items)
        self.nonce = nonce or secrets.token_hex(6)
        self._by_label = {i.label: i for i in self.items}

    # ------------------------------------------------------------------ construction
    @classmethod
    def build(
        cls,
        groups: Sequence[tuple[str, Sequence[RetrievedPassage | Passage]]],
        max_passages: int = 8,
        max_chars: int = 9000,
        nonce: str | None = None,
    ) -> EvidenceSet:
        """Combine ``(origin, passages)`` groups. Budgets are split evenly between groups; passages
        are taken in the given (ranked) order, skipping duplicates, until a budget is exhausted."""
        groups = [(o, list(ps)) for o, ps in groups if ps]
        items: list[EvidenceItem] = []
        seen_ids: set[str] = set()
        seen_text: set[str] = set()
        if not groups:
            return cls([], nonce)
        per_n = max(1, max_passages // len(groups))
        per_chars = max(1, max_chars // len(groups))
        for origin, passages in groups:
            used_n = used_chars = 0
            for entry in passages:
                passage = entry.passage if isinstance(entry, RetrievedPassage) else entry
                score = entry.score if isinstance(entry, RetrievedPassage) else None
                key = normalize_for_match(passage.text)
                if passage.passage_id in seen_ids or key in seen_text:
                    continue
                if used_n >= per_n or used_chars + len(passage.text) > per_chars:
                    continue
                seen_ids.add(passage.passage_id)
                seen_text.add(key)
                used_n += 1
                used_chars += len(passage.text)
                items.append(EvidenceItem(f"E{len(items) + 1}", passage, origin, score))
        return cls(items, nonce)

    # ------------------------------------------------------------------ access
    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[EvidenceItem]:
        return iter(self.items)

    @property
    def labels(self) -> list[str]:
        return [i.label for i in self.items]

    def get(self, label: str) -> EvidenceItem | None:
        return self._by_label.get(label)

    def text_for(self, labels: Sequence[str]) -> str:
        return "\n".join(self._by_label[label].passage.text for label in labels if label in self._by_label)

    def total_chars(self) -> int:
        return sum(len(i.passage.text) for i in self.items)

    # ------------------------------------------------------------------ prompt formatting
    def to_prompt_block(self) -> str:
        """Fenced block for the prompt. The nonce is removed from the text so it cannot be forged."""
        lines = [f"<<<EVIDENCE_START {self.nonce}>>>"]
        for item in self.items:
            p = item.passage
            safe_title = (p.title or "").replace(self.nonce, "")
            safe_text = p.text.replace(self.nonce, "")
            lines.append(f"[{item.label}] origin={item.origin} | {item.describe()} | title: {safe_title}")
            lines.append(safe_text)
            lines.append("")
        lines.append(f"<<<EVIDENCE_END {self.nonce}>>>")
        return "\n".join(lines)
