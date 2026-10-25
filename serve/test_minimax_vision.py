"""MiniMax vision in the server: the template's markers, token counts, and the picture path the engine is given.

No model, no network, no engine: the MiniMax tokenizer is test_minimax's synthetic byte-level BPE, the engine is
MockEngine, and the picture is a small PNG made in memory.  The engine's own ViT is checked by tools/minimax
against the HF reference; this file checks the server's side of that contract.
"""
from __future__ import annotations

import base64
import io
import json
import threading
from pathlib import Path

import pytest
from PIL import Image

from serve.frontend import anthropic_to_messages, images_of, openai_to_messages
from serve.minimax import MiniMaxTemplate, MiniMaxTokenizer
from serve.server import EngineVision, MockEngine, Service
from serve.test_minimax import _write_tokenizer


def png_bytes(w=640, h=480):
    b = io.BytesIO()
    Image.new("RGB", (w, h), (200, 30, 30)).save(b, format="PNG")
    return b.getvalue()


@pytest.fixture()
def tok(tmp_path):
    _write_tokenizer(tmp_path)
    return MiniMaxTokenizer(tmp_path)


def _service(tok, ctx=32768):
    eng = MockEngine(tok, "ok", max_context=ctx)
    svc = Service(eng, tok, MiniMaxTemplate(), model_name="minimax-m3.1-flash", format="minimax_m3",
                  engine_vision=EngineVision({"max_tokens": 576}))
    return svc, eng


def test_template_writes_the_checkpoint_image_markers(tok):
    out = MiniMaxTemplate().render([{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image", "source": "x.png"}]}])
    assert out.count(']<]start of image[>[') == 1
    assert out.count(']<]end of image[>[') == 1
    assert out.count(']<]image[>[') == 1
    assert out.index(']<]start of image[>[') < out.index(']<]image[>[') < out.index(']<]end of image[>[')


# The checkpoint's own chat_template.jinja (the one `sparsa.py download tokenizer` fetches) writes a bare image
# token and leaves the wrapping to the HF processor.  The server has to produce the same prompt for it.
CHECKPOINT_STYLE = """{%- for message in messages -%}
{{- message.role }}:
{%- if message.content is string -%}
{{- message.content }}
{%- else -%}
{%- for tr in message.content -%}
{%- if tr is mapping and tr.type == 'text' -%}
{{- tr.text }}
{%- elif tr is mapping and tr.type == 'image' -%}
]<]image[>[
{%- endif -%}
{%- endfor -%}
{%- endif -%}
{{ '\n' }}
{%- endfor -%}
"""


def test_a_template_with_a_bare_image_token_gets_the_triple(tmp_path):
    path = tmp_path / "checkpoint.jinja"
    path.write_text(CHECKPOINT_STYLE, encoding="utf-8")
    tpl = MiniMaxTemplate(path)
    out = tpl.render([{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image", "source": "x.png"}]}])
    assert out.count(']<]start of image[>[') == 1
    assert out.count(']<]image[>[') == 1
    assert out.count(']<]end of image[>[') == 1


def test_the_bare_image_template_still_prepares_the_picture(tok, tmp_path):
    path = tmp_path / "checkpoint.jinja"
    path.write_text(CHECKPOINT_STYLE, encoding="utf-8")
    tpl = MiniMaxTemplate(path)
    eng = MockEngine(tok, "ok", max_context=32768)
    svc = Service(eng, tok, tpl, model_name="minimax-m3.1-flash", format="minimax_m3",
                  engine_vision=EngineVision({"max_tokens": 576}))
    src = "data:image/png;base64," + base64.b64encode(png_bytes()).decode()
    messages, tools, kwargs = openai_to_messages({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": src}}]}]})
    ids, _, _ = svc.prepare(messages, tools, kwargs, max_new=8)
    assert ids.count(tok.ids[']<]image[>[']) == 391



def test_token_count_matches_the_checkpoint_smart_resize():
    # 672x672 is the 576-token maximum; OCR's sizes round to 28-pixel blocks, then merge 2x2.
    assert EngineVision.tokens_for(672, 672) == 576
    assert EngineVision.tokens_for(640, 480) == 391      # 476x644 -> 34x46 patches -> 391
    assert EngineVision.tokens_for(500, 700) == 450      # 504x700 -> 36x50 -> 450
    assert EngineVision.tokens_for(100, 100) == 16       # round to 112x112 (8x8 patches, merged 2x2)


def test_openai_image_url_reaches_the_prompt_as_391_tokens(tok):
    svc, eng = _service(tok)
    src = "data:image/png;base64," + base64.b64encode(png_bytes()).decode()
    messages, tools, kwargs = openai_to_messages({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": src}}]}]} )
    assert images_of(messages) == [src]
    ids, thinking, max_new = svc.prepare(messages, tools, kwargs, max_new=8)
    pad = tok.ids[']<]image[>[']
    start = tok.ids[']<]start of image[>[']
    end = tok.ids[']<]end of image[>[']
    assert ids.count(pad) == 391
    i = ids.index(start)
    assert ids[i + 1:i + 1 + 391] == [pad] * 391
    assert ids[i + 392] == end
    assert svc.embeddings.path.startswith("img=")
    assert svc.embeddings.path.endswith(".img")
    assert thinking is True and max_new == 8


def test_anthropic_base64_image_uses_the_same_path(tok):
    svc, eng = _service(tok)
    data = base64.b64encode(png_bytes(500, 700)).decode()
    messages, tools, kwargs = anthropic_to_messages({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "what does the card say?"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}]}]})
    ids, _, _ = svc.prepare(messages, tools, kwargs, max_new=8)
    assert ids.count(tok.ids[']<]image[>[']) == 450
    assert svc.embeddings.path.startswith("img=")


def test_the_engine_is_given_img_paths_and_not_an_embeddings_file(tok):
    svc, eng = _service(tok)
    src = "data:image/png;base64," + base64.b64encode(png_bytes()).decode()
    messages, tools, kwargs = openai_to_messages({"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": src}]}]})
    ids, _, _ = svc.prepare(messages, tools, kwargs, max_new=4)
    path = svc.embeddings.path
    assert path.startswith("img=") and ".sve" not in path
    files = path[len("img="):].split(",")
    assert len(files) == 1 and Path(files[0]).is_file() and files[0].endswith(".img")
    # the GENI head a run builds is "GENI <max> <keys> img=<paths> <ids>" (Engine.generate)
    assert f" {path} " in f"GENI 4 temperature=0 {path} " + " ".join(map(str, ids))


def test_engine_vision_refuses_over_the_limit(tmp_path):
    v = EngineVision({"max_tokens": 10})
    with pytest.raises(ValueError, match="over the limit"):
        v.encode(png_bytes())
    v.close()


def test_text_only_still_works_without_a_picture(tok):
    svc, _ = _service(tok)
    ids, _, _ = svc.prepare([{"role": "user", "content": "hello"}], [], {}, max_new=4)
    assert tok.ids[']<]image[>['] not in ids
    assert svc.embeddings.path is None
