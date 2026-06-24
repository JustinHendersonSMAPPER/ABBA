"""Tests for the Provenance dataclass and TrustTier enum."""

import pytest

from abba.provenance import Provenance, TrustTier


def test_valid_ingested_fact_has_no_confidence() -> None:
    p = Provenance(
        entity_type="cross_reference",
        entity_id="42",
        source="TSK",
        trust_tier=TrustTier.AUTHORITATIVE,
        trust_rationale="Public-domain scholarly compilation (~1880), widely used standard.",
        pipeline_version="0.1.0",
    )
    assert p.confidence is None
    assert p.trust_tier is TrustTier.AUTHORITATIVE


def test_llm_output_requires_confidence() -> None:
    with pytest.raises(ValueError, match="confidence is missing"):
        Provenance(
            entity_type="cross_reference",
            entity_id="42",
            source="ollama",
            trust_tier=TrustTier.GENERATED,
            trust_rationale="Explanation grounded in 2 shared Strong's numbers.",
            pipeline_version="0.1.0",
            generated_by="qwen3.5:cloud",
        )


def test_confidence_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
        Provenance(
            entity_type="x",
            entity_id="1",
            source="ollama",
            trust_tier=TrustTier.GENERATED,
            trust_rationale="r",
            pipeline_version="0.1.0",
            generated_by="qwen",
            confidence=1.5,
        )


def test_blank_rationale_rejected() -> None:
    with pytest.raises(ValueError, match="trust_rationale"):
        Provenance(
            entity_type="x",
            entity_id="1",
            source="TSK",
            trust_tier=TrustTier.AUTHORITATIVE,
            trust_rationale="",
            pipeline_version="0.1.0",
        )


def test_grounding_json_is_stable() -> None:
    p = Provenance(
        entity_type="cross_reference",
        entity_id="1",
        source="ollama",
        trust_tier=TrustTier.GENERATED,
        trust_rationale="r",
        pipeline_version="0.1.0",
        generated_by="qwen",
        confidence=0.8,
        grounding={"b": 2, "a": 1},
    )
    assert p.grounding_json() == '{"a": 1, "b": 2}'
