"""LLM validation methodology documentation and metadata.

Documents the methodology used for LLM-based concept validation,
including model versions, theological limitations, and reproducibility.
"""

VALIDATION_METHODOLOGY = {
    "title": "ABBA LLM Concept Validation Methodology",
    "version": "1.0",
    "overview": (
        "ABBA uses Large Language Models (LLMs) via Ollama for semantic concept validation. "
        "This process validates whether verses matched by Strong's concordance actually express "
        "the mapped concept. LLMs are used ONLY at build time; no LLM is needed at search time."
    ),
    "process": [
        {
            "step": 1,
            "name": "Lexical Matching",
            "description": (
                "Strong's numbers from concept definitions are matched against the words table. "
                "This is purely lexicographic — no AI involved. High precision but may miss "
                "verses that express a concept without using the specific mapped terms."
            ),
        },
        {
            "step": 2,
            "name": "LLM Validation",
            "description": (
                "Each lexically-matched verse is submitted to one or more Ollama models with "
                "a structured prompt asking: 'Does this verse genuinely express the concept X, "
                "or is the word being used in a different sense?' Models respond with a "
                "confidence score (0.0-1.0) and reasoning."
            ),
        },
        {
            "step": 3,
            "name": "Consensus Scoring",
            "description": (
                "When multiple models are configured, their scores are averaged. A verse must "
                "meet the consensus_threshold (default 0.7) to be included. This multi-model "
                "approach reduces individual model bias."
            ),
        },
    ],
    "models_tested": [
        {
            "model": "llama3",
            "version": "8B",
            "notes": "Default model. Good balance of speed and theological awareness.",
        },
        {
            "model": "llama3:70b",
            "version": "70B",
            "notes": "Higher accuracy but slower. Recommended for final validation passes.",
        },
        {
            "model": "mistral",
            "version": "7B",
            "notes": "Alternative model. Useful as a second opinion in consensus mode.",
        },
    ],
    "theological_limitations": [
        "LLMs reflect training data biases, which skew toward Western Protestant theology.",
        "Eschatological concepts (rapture, millennium) receive inconsistent treatment.",
        "Inter-testamental and Second Temple Judaism concepts may be under-represented.",
        "Catholic/Orthodox theological distinctives may not be fully captured.",
        "Hebrew wordplay and literary structures are often missed by LLMs.",
        "LLMs may conflate systematic theology with biblical theology.",
    ],
    "reproducibility": {
        "determinism": (
            "Ollama models with temperature=0 produce near-deterministic output, but "
            "exact reproducibility depends on model version, quantization, and hardware."
        ),
        "versioning": (
            "All concept validation results should record: model name, model version hash, "
            "validation date, and prompt template version. This enables audit trails."
        ),
        "recommendations": [
            "Pin Ollama model versions for production validation runs.",
            "Store raw LLM responses alongside confidence scores for auditing.",
            "Re-validate periodically as models improve.",
            "Cross-check high-confidence results against scholarly commentaries.",
        ],
    },
    "framing_guidelines": [
        "Present meaning-richness as 'the original adds depth' — never 'your Bible is wrong'.",
        "Flag confessional readings (Trinity, original sin) as interpretive tradition.",
        "Distinguish descriptive passages from prescriptive commands.",
        "Always show speaker attribution to prevent misattribution.",
        "Include surrounding context to discourage proof-texting.",
    ],
}


def get_methodology_summary() -> str:
    """Return a human-readable summary of the validation methodology."""
    m = VALIDATION_METHODOLOGY
    title = str(m["title"])
    version = str(m["version"])
    overview = str(m["overview"])
    lines = [
        f"# {title} (v{version})",
        "",
        overview,
        "",
        "## Process",
    ]
    process_steps = m["process"]
    if isinstance(process_steps, list):
        for step in process_steps:
            lines.append(f"  {step['step']}. **{step['name']}**: {step['description']}")
    lines.append("")
    lines.append("## Theological Limitations")
    limitations = m["theological_limitations"]
    if isinstance(limitations, list):
        for lim in limitations:
            lines.append(f"  - {lim}")
    lines.append("")
    lines.append("## Framing Guidelines")
    guidelines = m["framing_guidelines"]
    if isinstance(guidelines, list):
        for guide in guidelines:
            lines.append(f"  - {guide}")
    return "\n".join(lines)
