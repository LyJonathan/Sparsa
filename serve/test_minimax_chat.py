"""MiniMax-M3.1-Flash chat template and reasoning, through both APIs.

The golden prompts in serve/fixtures/minimax_chat_golden.json were rendered by running the checkpoint's own
chat_template.jinja through jinja2 with transformers' settings.  The API tests use the checkpoint's
<mm:think>...</mm:think> tags over the OpenAI and Anthropic routes, streaming and not.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from serve.minimax import MiniMaxTemplate
from serve.server import MockEngine, Service, serve
from serve.test_minimax import _write_tokenizer

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "serve/fixtures/minimax_chat_golden.json").read_text(encoding="utf-8"))
CHECKPOINT = Path(__file__).resolve().parents[2] / "models" / "MiniMax-M3.1-Flash" / "chat_template.jinja"

THOUGHT = "work out two plus two"
ANSWER = "The answer is 4."
SCRIPT = f"<mm:think>{THOUGHT}</mm:think>{ANSWER}"


@pytest.fixture()
def tok(tmp_path):
    _write_tokenizer(tmp_path)
    from serve.minimax import MiniMaxTokenizer
    return MiniMaxTokenizer(tmp_path)


class MiniMaxEngine(MockEngine):
    def __init__(self, tokenizer, text):
        super().__init__(tokenizer, "", max_context=8192)
        self.tok, self.text, self.prompts = tokenizer, text, []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        for t in (self.tok.encode(self.text, parse_special=True) + [self.tok.eos_id])[:max_new]:
            if cancel.is_set():
                return
            yield t


class Harness:
    def __init__(self, tok):
        self.tok = tok
        self.engine = MiniMaxEngine(tok, SCRIPT)
        self.svc = Service(self.engine, tok, MiniMaxTemplate(), model_name="minimax-m3.1-flash",
                           format="minimax_m3")
        self.svc.model_arch = "minimax_m3"
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode()

    def events(self, path, body):
        raw = self.post(path, body)
        return [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]


def test_golden_prompts_match_the_checkpoints_jinja():
    for case in GOLDEN:
        got = MiniMaxTemplate().render(case["messages"], tools=case["tools"], **case["kwargs"])
        assert got == case["expected"], case["name"]


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="the checkpoint's chat_template.jinja is not in this checkout")
def test_the_checkpoints_own_template_renders_the_same_text_prompts():
    tpl = MiniMaxTemplate(CHECKPOINT)
    for case in GOLDEN:
        assert tpl.render(case["messages"], tools=case["tools"], **case["kwargs"]) == case["expected"], case["name"]


def test_openai_non_stream_reports_mm_think_as_reasoning(tok):
    h = Harness(tok)
    try:
        body = json.loads(h.post("/v1/chat/completions", {
            "model": "m", "messages": [{"role": "user", "content": "2+2?"}], "max_tokens": 64}))
        msg = body["choices"][0]["message"]
        assert msg["reasoning_content"] == THOUGHT
        assert msg["content"] == ANSWER
        assert "<mm:think>" not in msg["content"] and "</mm:think>" not in msg["content"]
    finally:
        h.close()


def test_openai_stream_reports_mm_think_as_reasoning(tok):
    h = Harness(tok)
    try:
        evs = h.events("/v1/chat/completions", {
            "model": "m", "messages": [{"role": "user", "content": "2+2?"}], "max_tokens": 64, "stream": True})
        reasoning = "".join(e["choices"][0]["delta"].get("reasoning_content") or "" for e in evs)
        content = "".join(e["choices"][0]["delta"].get("content") or "" for e in evs)
        assert reasoning == THOUGHT
        assert content == ANSWER
        assert "<mm:think>" not in reasoning + content
    finally:
        h.close()


def test_anthropic_non_stream_has_a_thinking_block(tok):
    h = Harness(tok)
    try:
        body = json.loads(h.post("/v1/messages", {
            "model": "m", "max_tokens": 64, "messages": [{"role": "user", "content": "2+2?"}]}))
        blocks = body["content"]
        assert [b["type"] for b in blocks] == ["thinking", "text"]
        assert blocks[0]["thinking"] == THOUGHT
        assert blocks[1]["text"] == ANSWER
    finally:
        h.close()


def test_anthropic_stream_has_thinking_deltas(tok):
    h = Harness(tok)
    try:
        evs = h.events("/v1/messages", {
            "model": "m", "max_tokens": 64, "stream": True,
            "messages": [{"role": "user", "content": "2+2?"}]})
        starts = [e["content_block"] for e in evs if e["type"] == "content_block_start"]
        thinking = "".join(e["delta"].get("thinking", "") for e in evs if e["type"] == "content_block_delta")
        text = "".join(e["delta"].get("text", "") for e in evs if e["type"] == "content_block_delta")
        assert starts[0]["type"] == "thinking"
        assert starts[1]["type"] == "text"
        assert thinking == THOUGHT
        assert text == ANSWER
    finally:
        h.close()


def test_thinking_from_earlier_turns_is_kept_in_the_prompt(tok):
    h = Harness(tok)
    try:
        h.post("/v1/chat/completions", {"model": "m", "max_tokens": 64, "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "old answer", "reasoning_content": "prior thought"},
            {"role": "user", "content": "now?"}]})
        prompt = h.tok.decode(h.engine.prompts[-1])
        assert "<mm:think>prior thought</mm:think>old answer" in prompt

        h.post("/v1/messages", {"model": "m", "max_tokens": 64, "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "earlier reasoning", "signature": ""},
                {"type": "text", "text": "old answer"}]},
            {"role": "user", "content": "now?"}]})
        prompt = h.tok.decode(h.engine.prompts[-1])
        assert "<mm:think>earlier reasoning</mm:think>old answer" in prompt
    finally:
        h.close()
