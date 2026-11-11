"""Run the pinned LFM3 Jinja template and reference tokenizer on plain text."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

CASES = [
    [{"role": "user", "content": "Hi"}],
    [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "Café ☕"}],
    [{"role": "user", "content": "A"}, {"role": "assistant", "content": "<think>hidden</think>\u00a0Yes  "},
     {"role": "user", "content": "B"}],
    [{"role": "system", "content": ""}, {"role": "user", "content": "Hello"},
     {"role": "assistant", "reasoning": "one step", "content": "Answer"}],
    [{"role": "user", "content": ""}, {"role": "assistant", "thinking": None, "content": "<think>x</think>y"}],
]
RAW = ["  leading space\nCafé ☕\n", "🙂  你好\n", "<|startoftext|>A<|im_end|>"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ["directory", "reference_site", "output"]:
        parser.add_argument("--" + flag.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference_site))
    from tokenizers import Tokenizer
    from transformers.utils.chat_template_utils import _compile_jinja_template
    import tokenizers
    import transformers
    assert tokenizers.__version__ == "0.23.1" and transformers.__version__ == "5.20.2"
    directory = args.directory
    template = (directory / "chat_template.jinja").read_bytes()
    tok = (directory / "tokenizer.json").read_bytes()
    assert hashlib.sha256(template).hexdigest() == "f45005c50cd2a97329722bf3a2dee9f16c04cdf825a589b280fb7e99e6827818"
    assert hashlib.sha256(tok).hexdigest() == "8f7ed4182b60c4ade9645e2f98b8f2141961bfd71c783e15daaa89e95ae66338"
    engine = _compile_jinja_template(template.decode("utf-8"))
    tokenizer = Tokenizer.from_str(tok.decode("utf-8"))
    tokenizer.no_padding()
    rows = []
    for case in CASES:
        render = engine.render(messages=case, tools=None, bos_token="<|startoftext|>", add_generation_prompt=True)
        rows.append(dict(messages=case, text=render, ids=tokenizer.encode(render, add_special_tokens=False).ids))
    output = dict(
        chats=rows,
        raw=[dict(text=text, ids=tokenizer.encode(text, add_special_tokens=False).ids) for text in RAW],
        template_sha256=hashlib.sha256(template).hexdigest(),
        tokenizer_sha256=hashlib.sha256(tok).hexdigest(),
    )
    for row in output["raw"]:
        row["decoded"] = tokenizer.decode(row["ids"], skip_special_tokens=True)
    args.output.open("x").write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
