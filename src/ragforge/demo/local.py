"""Bounded generation with a pinned local instruction model."""

import re
from collections.abc import Sequence
from importlib import import_module
from typing import Any

from ragforge.domain.models import Citation, ProviderAnswer
from ragforge.generation.citations import validate_citation_references

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def cite_verbatim(text: str, citations: Sequence[Citation]) -> str:
    """Attach evidence when the generated words occur in exactly one passage."""
    phrase = text.strip().rstrip(".!?")
    if len(phrase.split()) < 2:
        raise ValueError("generated answer is too short to locate")
    pattern = re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", re.IGNORECASE)
    matches = [c for c in citations if pattern.search(c.quote)]
    if len(matches) != 1:
        raise ValueError("generated answer lacks a unique source match")
    citation_id = matches[0].citation_id
    answer = ProviderAnswer(text=f"{text} [{citation_id}]", citation_ids=(citation_id,))
    validate_citation_references(answer, citations)
    return answer.text


class LocalGenerator:
    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None

    def answer(self, question: str, citations: Sequence[Citation]) -> str:
        torch = import_module("torch")
        transformers = import_module("transformers")

        if self.model is None:
            options = dict(revision=REVISION, trust_remote_code=False, local_files_only=True)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, **options)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                MODEL, **options, use_safetensors=True, dtype="auto"
            ).to("mps" if torch.backends.mps.is_available() else "cpu")
            self.model.eval()
        sources = "\n".join(f"[{c.citation_id}] {c.quote}" for c in citations)
        messages = [
            {
                "role": "system",
                "content": "Copy the one complete sentence from the sources that answers "
                "the question. Output that sentence exactly, without any labels, citation "
                "markers or other words. If no sentence answers the question, output Unknown.",
            },
            {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        if encoded.input_ids.shape[1] > 1024:
            raise ValueError("input exceeds context budget")
        with torch.inference_mode():
            output = self.model.generate(
                **encoded.to(self.model.device), max_new_tokens=72, do_sample=False
            )
        text = self.tokenizer.decode(
            output[0, encoded.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        return cite_verbatim(text, citations)
