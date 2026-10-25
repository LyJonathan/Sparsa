"""Tests for the MiniMax format front end (serve/minimax.py).

No model, no network, no HF runtime: the tokenizer is a small synthetic byte-level BPE with the released tokenizer's
shape (ByteLevel pre-tokenizer, added tokens, eos [e~[).  The template is the checkpoint's Jinja file.
"""
from __future__ import annotations

import json

import pytest

from serve.minimax import (CALL_END, CALL_START, NS, THINK_END, MiniMaxOutputParser, MiniMaxTemplate,
                           MiniMaxTokenizer, parse_tool_calls)
from tools.strata_tokenizer import BYTE_TO_UNICODE


def _write_tokenizer(directory):
    vocab = {BYTE_TO_UNICODE[b]: b for b in range(256)}
    merges = []
    next_id = 256

    def add(word):
        nonlocal next_id
        parts = list(''.join(BYTE_TO_UNICODE[b] for b in word.encode('utf-8')))
        while len(parts) > 1:
            pair = (parts[0], parts[1])
            merged = pair[0] + pair[1]
            if merged not in vocab:
                vocab[merged] = next_id
                next_id += 1
                merges.append([pair[0], pair[1]])
            parts[:2] = [merged]
        if len(parts) == 1 and parts[0] not in vocab:
            vocab[parts[0]] = next_id
            next_id += 1
        return parts[0]

    for word in ("hello", " world", "hi"):
        add(word)

    added = ["]~b]", "[e~[", "]~!b[", "<mm:think>", "</mm:think>", "<tool_call>", "</tool_call>", NS,
             ']<]image[>[', ']<]start of image[>[', ']<]end of image[>[']
    added = [(text, next_id + i) for i, text in enumerate(added)]
    data = {
        "version": "1.0",
        "added_tokens": [{"id": i, "content": t, "special": True, "single_word": False, "lstrip": False,
                          "rstrip": False, "normalized": False} for t, i in added],
        # The released tokenizer.json's shape: NFC, and a Split that keeps only its matches before ByteLevel.
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "behavior": "Removed", "invert": True,
             "pattern": {"Regex": r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"}},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "vocab": vocab, "merges": merges, "unk_token": None, "byte_fallback": False},
    }
    (directory / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
    (directory / "tokenizer_config.json").write_text(json.dumps({"eos_token": "[e~[", "bos_token": "]~b]"}),
                                                    encoding="utf-8")
    return data


@pytest.fixture()
def tok(tmp_path):
    _write_tokenizer(tmp_path)
    return MiniMaxTokenizer(tmp_path)


def test_tokenizer_bpe_and_round_trip(tok):
    assert tok.eos_id == tok.ids["[e~["]
    assert tok.bos_id == tok.ids["]~b]"]
    assert tok.encode("hello", parse_special=True) == [tok.ids["hello"]]
    assert tok.encode("hi", parse_special=True) == [tok.ids["hi"]]
    text = "hello, world! h\u00e9llo \U0001f642\n"
    assert tok.decode(tok.encode(text, parse_special=True)) == text


def test_tokenizer_added_tokens(tok):
    assert tok.encode("[e~[", parse_special=True) == [tok.ids["[e~["]]
    assert tok.encode("]~b]", parse_special=True) == [tok.ids["]~b]"]]
    assert tok.encode(NS + "<mm:think>", parse_special=True) == [tok.ids[NS], tok.ids["<mm:think>"]]


def test_tokenizer_normalizes_to_nfc(tok):
    """The released tokenizer.json normalizes to NFC, so a decomposed prompt is the composed prompt's ids."""
    assert tok.encode("e\u0301", parse_special=True) == tok.encode("\u00e9", parse_special=True)
    assert tok.decode(tok.encode("e\u0301", parse_special=True)) == "\u00e9"


def test_template_renders_roles_thinking_and_generation_prompt():
    out = MiniMaxTemplate().render([{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}])
    assert out.startswith("]~!b[]~b]system\n")
    assert "]~b]developer\nSYS[e~[\n" in out
    assert "]~b]user\nhi[e~[\n" in out
    assert "<thinking_instructions>" in out
    assert out.endswith("]~b]ai\n")


def test_template_renders_tools_in_the_form_the_checkpoint_reads():
    tools = [{"name": "get_weather", "description": "weather", "parameters": {"type": "object", "properties": {}}}]
    out = MiniMaxTemplate().render([{"role": "user", "content": "x"}], tools=tools)
    assert "<tools>" in out
    assert '"get_weather"' in out
    assert NS + '<invoke name="tool-name-1">' in out


def test_template_renders_reasoning_and_tool_calls_in_history():
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer", "reasoning_content": "why",
         "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1, "b": True}}}]},
        {"role": "tool", "content": "ok"},
        {"role": "user", "content": "again"},
    ]
    out = MiniMaxTemplate().render(messages)
    assert "<mm:think>why</mm:think>answer" in out
    assert NS + '<invoke name="f">' in out
    assert NS + "<a>1" in out
    assert NS + "<b>true" in out
    assert "<response>ok</response>" in out


def test_parser_reasoning_then_content_across_deltas():
    text = "<mm:think>plan</mm:think>the answer"
    parser = MiniMaxOutputParser(thinking=True)
    events = []
    for ch in text:
        events += parser.feed(ch)
    events += parser.finish()
    assert "".join(e.text for e in events if e.kind == "reasoning") == "plan"
    assert "".join(e.text for e in events if e.kind == "content") == "the answer"
    assert not any(e.kind == "tool_call" for e in events)


def test_parser_strips_the_namespace_token_from_text():
    parser = MiniMaxOutputParser(thinking=False)
    events = []
    for ch in f"hello {NS}world":
        events += parser.feed(ch)
    events += parser.finish()
    assert "".join(e.text for e in events if e.kind == "content") == "hello world"


def test_parser_tool_call_with_namespace_and_param_types():
    text = (f"<mm:think>t</mm:think>I will call.\n{CALL_START}\n{NS}<invoke name=\"get_weather\">\n"
            f"{NS}<city>Paris</city>\n{NS}<days>2</days>\n{NS}<flags><item>a</item><item>b</item></flags>\n"
            f"{NS}<opts><unit>c</unit></opts>\n{NS}</invoke>\n{CALL_END}")
    parser = MiniMaxOutputParser(thinking=True, tools=[{"name": "get_weather"}])
    events = []
    for i in range(0, len(text), 7):
        events += parser.feed(text[i:i + 7])
    events += parser.finish()
    calls = [e.call for e in events if e.kind == "tool_call"]
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == {"city": "Paris", "days": 2, "flags": ["a", "b"], "opts": {"unit": "c"}}
    assert "".join(e.text for e in events if e.kind == "content").strip() == "I will call."


def test_parser_two_invokes_in_one_block():
    text = (f"{CALL_START}{NS}<invoke name=\"a\"><x>1</x></invoke>"
            f"{NS}<invoke name=\"b\"><y>two</y></invoke>{CALL_END}")
    events = MiniMaxOutputParser(thinking=False).feed(text)
    calls = [e.call for e in events if e.kind == "tool_call"]
    assert [(c.name, c.arguments) for c in calls] == [("a", {"x": 1}), ("b", {"y": "two"})]


def test_parser_keeps_a_block_without_invoke_as_text():
    parser = MiniMaxOutputParser(thinking=False)
    events = parser.feed(f"{CALL_START}not a call{CALL_END}")
    events += parser.finish()
    assert parser.unread == 1
    assert "".join(e.text for e in events if e.kind == "content") == f"{CALL_START}not a call{CALL_END}"


def test_parse_tool_calls_rejects_duplicate_parameter():
    with pytest.raises(ValueError):
        parse_tool_calls(f"{NS}<invoke name=\"f\"><a>1</a><a>2</a></invoke>")


def test_parser_thinking_disabled_starts_in_content():
    parser = MiniMaxOutputParser(thinking=False)
    events = parser.feed("answer") + parser.finish()
    assert [(e.kind, e.text) for e in events] == [("content", "answer")]
    assert THINK_END not in "".join(e.text for e in events)


def test_parser_adaptive_direct_answer_is_content_not_reasoning():
    """The generation prompt is adaptive: with thinking on the model may answer without opening a think block."""
    parser = MiniMaxOutputParser(thinking=True)
    events = parser.feed("the answer") + parser.finish()
    assert [(e.kind, e.text) for e in events] == [("content", "the answer")]


def test_parser_adaptive_no_thinking_prefix_is_content():
    parser = MiniMaxOutputParser(thinking=True)
    events = []
    for ch in "</mm:think>the answer":
        events += parser.feed(ch)
    events += parser.finish()
    assert "".join(e.text for e in events if e.kind == "content") == "the answer"
    assert not any(e.kind == "reasoning" for e in events)


def test_parser_adaptive_think_tag_split_across_deltas():
    parser = MiniMaxOutputParser(thinking=True)
    events = []
    for piece in ("<mm:", "thi", "nk>plan", "</mm:think>", "answer"):
        events += parser.feed(piece)
    events += parser.finish()
    assert "".join(e.text for e in events if e.kind == "reasoning") == "plan"
    assert "".join(e.text for e in events if e.kind == "content") == "answer"
