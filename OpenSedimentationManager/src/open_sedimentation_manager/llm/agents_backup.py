from __future__ import annotations

import os
from typing import Optional


def summarize_text(
    text: str,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_tokens: int = 300,
    temperature: float = 0.2,
) -> str:
    """Summarize or rewrite text using the OpenAI SDK.

    Requires OPENAI_API_KEY to be set in the environment. The model can be provided via
    the parameter or the OSM_OPENAI_MODEL environment variable. Raises RuntimeError with
    a clear message if the SDK is not installed or credentials are missing.
    """
    if model is None:
        model = os.getenv("OSM_OPENAI_MODEL", "gpt-4o-mini")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover - import-time check
        raise RuntimeError(
            "OpenAI SDK not installed. Install the 'llm' extra: pip install -e .[llm]"
        ) from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it securely in your environment before enabling LLM features."
        )

    client = OpenAI(api_key=api_key)

    sys_prompt = system_prompt or (
        "You are a domain expert in hydrology, dredging, and environmental compliance. "
        "Write concise, technically accurate summaries that preserve units and numeric values."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": text},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content or ""
    return content.strip()

