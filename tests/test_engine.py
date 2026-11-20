from types import SimpleNamespace as NS

import httpx2
import openai
import pytest

from ohbotchat.config import load_config
from ohbotchat.engine import EngineError
from ohbotchat.engine_chain import ChainEngine
from ohbotchat.events import EventBus


def _resp(status):
    return httpx2.Response(status, request=httpx2.Request("POST", "http://x"))


class FakeClient:
    def __init__(self, flagged=(False, False), text="Cześć!", transcript="Cześć robot", fail=None):
        self.calls, self.flagged, self.text, self.transcript, self.fail = [], list(flagged), text, transcript, fail
        self.audio = NS(
            transcriptions=NS(create=self._stt), speech=NS(create=self._tts)
        )
        self.chat = NS(completions=NS(create=self._chat))
        self.moderations = NS(create=self._mod)

    def _go(self, name, kwargs):
        self.calls.append((name, kwargs))
        if self.fail and self.fail[0] == name:
            raise self.fail[1]

    async def _stt(self, **kw):
        self._go("stt", kw)
        return NS(text=self.transcript)

    async def _mod(self, **kw):
        self._go("moderation", kw)
        return NS(results=[NS(flagged=self.flagged.pop(0))])

    async def _chat(self, **kw):
        self._go("chat", kw)
        return NS(choices=[NS(message=NS(content=self.text))])

    async def _tts(self, **kw):
        self._go("tts", kw)
        return NS(content=b"\x01\x00" * 10)


def make(client):
    cfg = load_config(env={})
    bus = EventBus()
    return ChainEngine(client, cfg, bus), bus


async def stages(bus_q):
    out = []
    while not bus_q.empty():
        e = bus_q.get_nowait()
        out.append((e.data["name"], e.data["status"]))
    return out


async def test_happy_path_and_stage_order():
    client = FakeClient()
    eng, bus = make(client)
    q = bus.subscribe()
    reply = await eng.respond("Cześć robot", [{"role": "user", "content": "a"}])
    assert (reply.heard, reply.text, reply.pcm) == ("Cześć robot", "Cześć!", b"\x01\x00" * 10)
    names = [n for n, s in await stages(q) if s == "start"]
    assert names == ["moderation_in", "chat", "moderation_out", "tts"]
    chat = dict(client.calls)["chat"]
    assert "Ohbot" in chat["messages"][0]["content"]
    assert chat["messages"][1] == {"role": "user", "content": "a"}
    assert chat["messages"][-1] == {"role": "user", "content": "Cześć robot"}
    assert dict(client.calls)["tts"]["response_format"] == "pcm"


async def test_transcribe_wraps_wav_and_sets_polish():
    client = FakeClient()
    eng, bus = make(client)
    q = bus.subscribe()
    assert await eng.transcribe(b"\x00\x00" * 160) == "Cześć robot"
    kw = dict(client.calls)["stt"]
    assert kw["file"][1][:4] == b"RIFF"
    assert kw["languages"] == ["pl"]
    assert await stages(q) == [("stt", "start"), ("stt", "done")]


async def test_blank_transcript_is_empty():
    eng, _ = make(FakeClient(transcript="  "))
    with pytest.raises(EngineError) as e:
        await eng.transcribe(b"\x00\x00")
    assert e.value.kind == "empty"


async def test_moderated_input_never_calls_chat():
    client = FakeClient(flagged=(True,))
    eng, _ = make(client)
    with pytest.raises(EngineError) as e:
        await eng.respond("x", [])
    assert e.value.kind == "moderated"
    assert [n for n, _ in client.calls] == ["moderation"]


async def test_moderated_reply_never_reaches_tts():
    client = FakeClient(flagged=(False, True))
    eng, _ = make(client)
    with pytest.raises(EngineError) as e:
        await eng.respond("x", [])
    assert e.value.kind == "moderated"
    assert "tts" not in [n for n, _ in client.calls]


def _quota():
    return openai.RateLimitError(
        "quota", response=_resp(429), body={"error": {"code": "insufficient_quota", "type": "insufficient_quota"}}
    )


@pytest.mark.parametrize(
    "exc, kind",
    [
        (lambda: openai.APIConnectionError(request=httpx2.Request("POST", "http://x")), "network"),
        (lambda: openai.APITimeoutError(request=httpx2.Request("POST", "http://x")), "network"),
        (_quota, "quota"),
        (lambda: openai.RateLimitError("slow down", response=_resp(429), body={"error": {"code": "rate_limit_exceeded"}}), "other"),
        (lambda: openai.AuthenticationError("bad key", response=_resp(401), body=None), "auth"),
    ],
)
async def test_error_mapping(exc, kind):
    eng, _ = make(FakeClient(fail=("chat", exc())))
    with pytest.raises(EngineError) as e:
        await eng.respond("x", [])
    assert e.value.kind == kind
