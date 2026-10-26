"""serve/minimax.py - MiniMax-M3.1-Flash's tokenizer, chat template and output parser (P2a stub).

The upstream server is Qwen-shaped: `think` is ``</think>``, tool calls are
``<tool_call><function=NAME><parameter=P>VALUE</parameter></function></tool_call>`` and the tokenizer comes from a
pack's ``vocab.json``.  MiniMax uses a byte-level BPE from the checkpoint's ``tokenizer.json``, ``<mm:think>``
reasoning tags, and XML calls of the form::

    <tool_call><invoke name="tool"><p>value</p></invoke></tool_call>

(with an optional ``]<]minimax[>[`` namespace token before every element).  This module supplies the three pieces
behind the run config's ``"format": "minimax_m3"`` key; the Qwen classes in ``serve/frontend.py`` are untouched.

Not implemented here (P2b/P5): reasoning_effort semantics beyond passing the level to the template, streaming
tool-argument fragments, images.  The parser buffers a whole ``<tool_call>`` block and emits its calls together.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import regex

from serve.frontend import ChatTemplate, Event, OutputParser, ToolCall
from tools.strata_tokenizer import BYTE_TO_UNICODE, Tokenizer

NS = ']<]minimax[>['
THINK_BEGIN = '<mm:think>'
THINK_END = '</mm:think>'
CALL_START = '<tool_call>'
CALL_END = '</tool_call>'
IMAGE_TOKEN = ']<]image[>['
VISION_START = ']<]start of image[>['
VISION_END = ']<]end of image[>['
IMAGE_PAD_ID = 200025
EOS_ID = 200020          # [e~[ in the released tokenizer; also read from tokenizer_config.json below

# Hugging Face's ByteLevel pre-tokenizer, \p classes included: this is the split the released BPE was trained on.
BYTELEVEL_PATTERN = regex.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")

DEFAULT_TEMPLATE = Path(__file__).with_name('minimax_chat_template.jinja')

# A private-use marker that stands in for one image part while the template renders.  The rendered text is then
# rewritten to the vision triple, so a template that writes only `]<]image[>[` (the checkpoint's own) and one that
# writes the whole thing (this repository's) produce the same prompt.
_IMAGE_SLOT_TEXT = '\ue000minimax-img-%d\ue001'
_IMAGE_SLOT = re.compile('\ue000minimax-img-\\d+\ue001')


def _image_slots(messages):
    """(messages with each image part as a marker text part, whether any image was found).  The input is unchanged."""
    if not isinstance(messages, (list, tuple)):
        return messages, False
    out, n = [], 0
    for message in messages:
        content = message.get('content') if isinstance(message, dict) else None
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get('type') == 'image':
                    parts.append({'type': 'text', 'text': _IMAGE_SLOT_TEXT % n})
                    n += 1
                else:
                    parts.append(part)
            message = {**message, 'content': parts}
        out.append(message)
    return out, n > 0



class MiniMaxTokenizer(Tokenizer):
    """Read ``tokenizer.json`` (byte-level BPE) without the ``tokenizers`` or ``transformers`` packages."""

    def __init__(self, directory):
        directory = Path(directory)
        path = directory / 'tokenizer.json'
        if not path.exists():
            raise ValueError(f'MiniMax tokenizer.json is missing ({path})')
        data = json.loads(path.read_text(encoding='utf-8'))
        model = data.get('model') or {}
        if model.get('type') != 'BPE':
            raise ValueError(f"MiniMax requires a byte-level BPE tokenizer, got {model.get('type')!r}")
        # The released tokenizer.json normalizes to NFC (a Sequence holding it in some exports).  NFC is applied
        # per pre-tokenizer piece below, which is where it can change the bytes: the pattern keeps a letter and
        # its combining marks in one piece, so composition never crosses a split.
        normalizer = data.get('normalizer')
        if isinstance(normalizer, dict) and normalizer.get('type') == 'Sequence':
            parts = normalizer.get('normalizers') or []
            if len(parts) == 1:
                normalizer = parts[0]
        if normalizer in (None, {'type': 'Sequence', 'normalizers': []}):
            self.nfc = False
        elif isinstance(normalizer, dict) and normalizer.get('type') == 'NFC':
            self.nfc = True
        else:
            raise ValueError(f'MiniMax tokenizer has an unsupported normalizer ({normalizer!r}); only NFC is known')

        vocab = dict(model['vocab'])
        added = data.get('added_tokens') or []
        for token in added:
            if any(token.get(k) for k in ('single_word', 'lstrip', 'rstrip')):
                raise ValueError('unsupported MiniMax added-token matching options')
            if not isinstance(token.get('content'), str) or not isinstance(token.get('id'), int):
                raise ValueError('malformed MiniMax added_tokens entry')
            vocab[token['content']] = token['id']
        size = max(vocab.values()) + 1
        tokens = [None] * size
        for text, i in vocab.items():
            if tokens[i] is not None and tokens[i] != text:
                raise ValueError(f'tokenizer id {i} names two tokens: {tokens[i]!r} and {text!r}')
            tokens[i] = text
        if any(t is None for t in tokens):
            raise ValueError('MiniMax vocabulary ids are not contiguous; cannot tokenize')
        merges = [' '.join(m) if isinstance(m, list) else m for m in model['merges']]
        super().__init__(tokens, merges, pre='minimax_m3')

        # Every added token is a literal the prompt spells (]~b], [e~[, <mm:think>, ...).  As in Strata-DS's
        # DeepSeek tokenizer, they match with and without parse_special: the chat template writes them as text.
        self.special_tokens = {t['content']: t['id'] for t in added}
        self._special_re = self._alt(list(self.special_tokens))
        self._always_re = self._special_re
        self._added_bytes = {t['id']: t['content'].encode('utf-8') for t in added}
        # The template's own control literals.  `literal_tags` marks these inside user text so a file an agent
        # reads cannot open an image/tool/reasoning block (#150, #554).
        self.control_tokens = [IMAGE_TOKEN, VISION_START, VISION_END, THINK_BEGIN, THINK_END,
                               CALL_START, CALL_END, NS]
        self.max_special_len = max((len(t) for t in self.special_tokens), default=1)

        self.splits: list[regex.Pattern] = []
        self.byte_regex = True
        pre = data.get('pre_tokenizer')
        if pre is not None:
            parts = pre.get('pretokenizers', []) if pre.get('type') == 'Sequence' else [pre]
            if not parts:
                raise ValueError('MiniMax tokenizer has an empty pre_tokenizer')
            for part in parts:
                if part.get('type') == 'Split':
                    pattern = part.get('pattern') or {}
                    behavior, invert = part.get('behavior'), bool(part.get('invert'))
                    # The released tokenizer keeps ONLY what the pattern matches ("Removed" + invert, the
                    # GPT-4 split); "Isolated" keeps the matches and the text between them.
                    if 'Regex' not in pattern or (behavior, invert) not in (('Isolated', False), ('Removed', True)):
                        raise ValueError('unsupported MiniMax Split pre-tokenizer')
                    self.splits.append((regex.compile(pattern['Regex']), behavior == 'Removed'))
                    continue
                if part.get('type') != 'ByteLevel':
                    raise ValueError(f"unsupported MiniMax pre-tokenizer {part.get('type')!r}")
                if part.get('add_prefix_space'):
                    raise ValueError('MiniMax tokenizer with add_prefix_space is not supported')
                self.byte_regex = bool(part.get('use_regex', True))

        cfg_path = directory / 'tokenizer_config.json'
        cfg = json.loads(cfg_path.read_text(encoding='utf-8')) if cfg_path.exists() else {}
        eos = cfg.get('eos_token', '[e~[')
        eos = eos.get('content') if isinstance(eos, dict) else eos
        self.eos_id = self.special_tokens.get(eos, EOS_ID)
        bos = cfg.get('bos_token', ']~b]')
        bos = bos.get('content') if isinstance(bos, dict) else bos
        self.bos_id = self.special_tokens.get(bos, 200019)

    def _pieces(self, text: str):
        pieces = [text]
        for pattern, keep_only in self.splits:
            split = []
            for piece in pieces:
                if keep_only:
                    split.extend(m.group() for m in pattern.finditer(piece) if m.end() > m.start())
                    continue
                pos = 0
                for match in pattern.finditer(piece):
                    if match.start() > pos:
                        split.append(piece[pos:match.start()])
                    if match.end() > match.start():
                        split.append(match.group())
                    pos = match.end()
                if pos < len(piece):
                    split.append(piece[pos:])
            pieces = split
        for piece in pieces:
            if not piece:
                continue
            yield from (BYTELEVEL_PATTERN.findall(piece) if self.byte_regex else [piece])

    def _encode_plain(self, text: str):
        out: list[int] = []
        cache = self._piece_ids
        for piece in self._pieces(text):
            if self.nfc:
                import unicodedata
                piece = unicodedata.normalize('NFC', piece)
            got = cache.get(piece)
            if got is None:
                mapped = ''.join(BYTE_TO_UNICODE[b] for b in piece.encode('utf-8'))
                got = []
                for token in self._bpe(mapped):
                    i = self.ids.get(token)
                    if i is None:
                        raise KeyError(f'BPE produced a token outside the vocabulary: {token!r}')
                    got.append(i)
                if len(cache) < self.PIECE_CACHE_MAX:
                    cache[piece] = got
            out.extend(got)
        return out

    def token_bytes(self, i):
        if i in self._added_bytes:
            return self._added_bytes[i]
        return super().token_bytes(i)


class MiniMaxTemplate(ChatTemplate):
    """The checkpoint's Jinja template, with the server's tools in the shape that template reads."""

    def __init__(self, path: str | Path | None = None):
        super().__init__(path or DEFAULT_TEMPLATE)

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        # serve/frontend.py hands templates bare {"name", "description", "parameters"} dicts; MiniMax's template
        # (like the HF one it came from) reads tool.function.
        if tools:
            tools = [t if isinstance(t, dict) and 'function' in t else {'type': 'function', 'function': t}
                     for t in tools]
        kwargs = dict(kwargs)
        effort = kwargs.get('reasoning_effort')
        if effort == 'xhigh':
            kwargs['reasoning_effort'] = 'max'       # the card spells the top level "max"
        elif effort in ('minimal', 'none', 'off'):
            kwargs.pop('reasoning_effort', None)
        if kwargs.get('enable_thinking') is False:
            kwargs.pop('reasoning_effort', None)
        messages, slots = _image_slots(messages)
        text = super().render(messages, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs)
        if slots:
            # The checkpoint's own chat_template.jinja writes a bare `]<]image[>[` and leaves the wrapping to the HF
            # processor; serve/minimax_chat_template.jinja writes the whole triple.  Either way the prompt the engine
            # reads is the processor's: start, one pad per image token, end.
            text = _IMAGE_SLOT.sub(VISION_START + IMAGE_TOKEN + VISION_END, text)
        return text


_TAG = re.compile(r'<([A-Za-z_][\w.\-]*)>')
_PARAM = re.compile(r'<([A-Za-z_][\w.\-]*)>(.*?)</\1>', re.S)
_INVOKE = re.compile(r'<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>', re.S)


def _scalar(text: str):
    """A parameter's text as JSON's scalar when it is one, else the string itself (``true``, ``12``, ``"x"``)."""
    s = text.strip()
    if s.startswith(('{', '[')):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    if s in ('true', 'false', 'null') or re.fullmatch(r'-?\d+(\.\d+)?([eE][+-]?\d+)?', s):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    return text


def _value(body: str):
    """A parameter body: nested XML becomes a dict (or a list for repeated <item>), text becomes a scalar."""
    children = []
    for name, inner in _PARAM.findall(body):
        children.append((name, _value(inner)))
    if not children:
        return _scalar(body)
    if all(name == 'item' for name, _ in children):
        return [value for _, value in children]
    return {name: value for name, value in children}


def parse_tool_calls(body: str) -> list[ToolCall]:
    """Parse the inside of one ``<tool_call>`` block.  Raises ValueError on a malformed invoke."""
    body = body.replace(NS, '')
    calls = []
    for match in _INVOKE.finditer(body):
        name = match.group(1)
        args = {}
        for key, inner in _PARAM.findall(match.group(2)):
            if key in args:
                raise ValueError(f'duplicate parameter {key!r} in tool call {name!r}')
            args[key] = _value(inner)
        calls.append(ToolCall(name, args))
    return calls


class MiniMaxOutputParser:
    """Stream MiniMax text into reasoning/content events and whole tool calls.

    Feed deltas; a tag split across deltas is held back, so clients never see ``<mm:th`` or a partial call.  The
    ``tool_start``/``tool_args`` events are not emitted yet: a call is reported when its block closes (P5 can add
    the fragment stream).  ``pending``/``rescued``/``refused`` exist because Service.run reads those counters.
    """

    def __init__(self, thinking=True, tools=None, stream_tools=False, recover=False, declared_only=False):
        # MiniMax's generation prompt is adaptive: with thinking on, the model may still answer directly. Start in
        # `start` and let the first real tag decide; a reply that opens with `<mm:think>` is reasoning, one that
        # opens with anything else (or `</mm:think>`, the template's no-thinking prefix) is content.
        self.state = 'start' if thinking else 'content'
        self.buf = ''
        self.stream_tools = stream_tools
        self.schemas = {t.get('name') for t in tools or [] if isinstance(t, dict)}
        self.pending: list = []
        self.rescued = 0
        self.refused = 0
        self.unread = 0
        self.unread_names: list[str] = []

    def feed(self, delta):
        # The namespace token is a prompt delimiter the model sometimes repeats in its output (it did on
        # the full model, right after a tool call's text).  It is never part of an answer, so it is removed
        # before anything is shown; a partial one at a chunk's end is held, like a split tag.
        self.buf = (self.buf + delta).replace(NS, '')
        out: list[Event] = []
        while self.buf:
            if self.state == 'start':
                lead = len(self.buf) - len(self.buf.lstrip())
                tail = self.buf[lead:]
                if not tail:
                    break
                if tail.startswith(THINK_BEGIN):
                    self.buf = tail[len(THINK_BEGIN):]
                    self.state = 'reasoning'
                    continue
                if tail.startswith(THINK_END):
                    self.buf = self.buf[:lead] + tail[len(THINK_END):]
                    self.state = 'content'
                    continue
                if THINK_BEGIN.startswith(tail) or THINK_END.startswith(tail):
                    break                       # a tag split across deltas: wait for the rest
                self.state = 'content'
                continue
            if self.state in ('reasoning', 'content'):
                tags = ((THINK_BEGIN, THINK_END, CALL_START) if self.state == 'reasoning'
                        else (THINK_END, CALL_START))
                found = [(self.buf.find(tag), tag) for tag in tags if tag in self.buf]
                if not found:
                    hold = OutputParser._hold(self, self.buf, tags + (NS,))
                    safe = len(self.buf) - hold
                    if safe:
                        out.append(Event(self.state, self.buf[:safe]))
                        self.buf = self.buf[safe:]
                    break
                at, tag = min(found)
                if at:
                    out.append(Event(self.state, self.buf[:at]))
                self.buf = self.buf[at + len(tag):]
                if tag == THINK_END:
                    self.state = 'content'
                elif tag == CALL_START:
                    self.state = 'calls'
                # THINK_BEGIN: stay in reasoning, and do not show the tag itself
                continue
            # state == 'calls': buffer until the closing tag, then report every invoke together
            at = self.buf.find(CALL_END)
            if at < 0:
                break
            body, self.buf = self.buf[:at], self.buf[at + len(CALL_END):]
            try:
                calls = parse_tool_calls(body)
            except ValueError:
                self.unread += 1
                out.append(Event('content', CALL_START + body + CALL_END))
            else:
                if not calls:
                    self.unread += 1
                    out.append(Event('content', CALL_START + body + CALL_END))
                for call in calls:
                    out.append(Event('tool_call', call=call))
            self.state = 'content'
        return out

    def finish(self, reason=None):
        out = self.feed('')
        if self.buf:
            if self.state == 'start':                 # only whitespace, or a tag the reply ended inside
                out.append(Event('content', self.buf))
            elif self.state in ('reasoning', 'content'):
                out.append(Event(self.state, self.buf))
            else:
                # A length limit or cancel left a partial call: do not expose protocol fragments as text.
                self.unread += 1
            self.buf = ''
        return out


__all__ = ['MiniMaxTokenizer', 'MiniMaxTemplate', 'MiniMaxOutputParser', 'parse_tool_calls',
           'THINK_BEGIN', 'THINK_END', 'CALL_START', 'CALL_END', 'NS', 'EOS_ID']
