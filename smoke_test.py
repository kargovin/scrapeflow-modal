"""Exercise the deployed endpoint the same way ScrapeFlow's llm-worker does:
an OpenAI-compatible chat completion with a json_schema response_format.

    VLLM_ENDPOINT=https://<endpoint>/v1 VLLM_API_KEY=<token> python smoke_test.py
"""

import json
import os

from openai import OpenAI

MODEL = "meta-llama/Llama-3.1-8B-Instruct"

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "price": {"type": "number"},
    },
    "required": ["title", "price"],
}


def main() -> None:
    client = OpenAI(
        base_url=os.environ["VLLM_ENDPOINT"],
        api_key=os.environ["VLLM_API_KEY"],
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Extract data from:\n\nWidget Pro — now only $19.99!",
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "extraction", "schema": SCHEMA},
        },
    )
    print(json.loads(resp.choices[0].message.content))


if __name__ == "__main__":
    main()
