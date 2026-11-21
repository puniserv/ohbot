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
    def __init__(self, flagged=(False, False), text="Cześć!", transcript="Cześć robot", fail=None, finish="stop"):
        self.calls, self.flagged, self.text, self.transcript, self.fail = [], list(flagged), text, transcript, fail
        self.finish = finish
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
        return NS(choices=[NS(message=NS(content=self.text), finish_reason=self.finish)])

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
    safe = []
    with pytest.raises(EngineError) as e:
        await eng.respond("x", [], lambda: safe.append(1))
    assert e.value.kind == "moderated"
    assert [n for n, _ in client.calls] == ["moderation"]
    assert safe == []  # the transcript must not reach the screen


async def test_on_safe_runs_once_input_passed_moderation_before_chat():
    client = FakeClient()
    eng, _ = make(client)
    seen = []
    await eng.respond("x", [], lambda: seen.append([n for n, _ in client.calls]))
    assert seen == [["moderation"]]


async def test_moderated_reply_never_reaches_tts():
    client = FakeClient(flagged=(False, True))
    eng, _ = make(client)
    with pytest.raises(EngineError) as e:
        await eng.respond("x", [])
    assert e.value.kind == "moderated"
    assert "tts" not in [n for n, _ in client.calls]


def _quota():
    return openai.RateLimitError(
        "quota", response=_resp(429), body={"code": "insufficient_quota", "type": "insufficient_quota"}
    )


@pytest.mark.parametrize(
    "exc, kind",
    [
        (lambda: openai.APIConnectionError(request=httpx2.Request("POST", "http://x")), "network"),
        (lambda: openai.APITimeoutError(request=httpx2.Request("POST", "http://x")), "network"),
        (_quota, "quota"),
        (lambda: openai.RateLimitError("slow down", response=_resp(429), body={"code": "rate_limit_exceeded"}), "other"),
        (lambda: openai.AuthenticationError("bad key", response=_resp(401), body=None), "auth"),
    ],
)
async def test_error_mapping(exc, kind):
    eng, _ = make(FakeClient(fail=("chat", exc())))
    with pytest.raises(EngineError) as e:
        await eng.respond("x", [])
    assert e.value.kind == kind


async def test_synthesize_returns_pcm():
    client = FakeClient()
    eng, _ = make(client)
    assert await eng.synthesize("Hej") == b"\x01\x00" * 10
    assert dict(client.calls)["tts"]["input"] == "Hej"


async def test_reasoning_effort_passed_and_omitted_when_empty():
    client = FakeClient()
    eng, _ = make(client)
    await eng.respond("x", [])
    assert dict(client.calls)["chat"]["reasoning_effort"] == "none"
    client = FakeClient()
    eng, _ = make(client)
    eng._cfg.openai.reasoning_effort = ""
    await eng.respond("x", [])
    assert "reasoning_effort" not in dict(client.calls)["chat"]


async def test_synthesize_without_publish_is_silent():
    eng, bus = make(FakeClient())
    q = bus.subscribe()
    await eng.synthesize("Hej", publish=False)
    assert q.empty()


async def test_chat_asks_for_json_schema_and_parses_gesture():
    client = FakeClient(text='{"text": " Tak, to prawda! ", "gesture": "yes"}')
    eng, _ = make(client)
    reply = await eng.respond("x", [])
    assert (reply.text, reply.gesture) == ("Tak, to prawda!", "yes")
    fmt = dict(client.calls)["chat"]["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["properties"]["gesture"]["enum"] == ["none", "yes", "no", "surprised", "happy", "thinking"]
    assert [c[1]["input"] for c in client.calls if c[0] == "moderation"][-1] == "Tak, to prawda!"
    assert dict(client.calls)["tts"]["input"] == "Tak, to prawda!"


async def test_plain_text_chat_reply_is_used_as_is():
    reply = await make(FakeClient(text="Cześć!"))[0].respond("x", [])
    assert (reply.text, reply.gesture) == ("Cześć!", "none")


@pytest.mark.parametrize(
    "content,text",
    [
        ('{"text": "Cześć! Opowiem ci o', "Cześć! Opowiem ci o"),
        ('{"text": "Powiedział \\"hej\\" i po', 'Powiedział "hej" i po'),
        ('{"text": "Zażółć\\u0144', "Zażółć\u0144"),
        ('{"text": "Zażółć\\u01', "Zażółć"),
        ('{"text": "Koniec\\', "Koniec"),
        ('{ "text" : "Cała", "gest', "Cała"),
    ],
)
async def test_truncated_json_reply_recovers_the_text(content, text):
    client = FakeClient(text=content, finish="length")
    reply = await make(client)[0].respond("x", [])
    assert (reply.text, reply.gesture) == (text, "none")
    assert "{" not in reply.text and "{" not in dict(client.calls)["tts"]["input"]


@pytest.mark.parametrize("content", ['{"gesture": "yes"}', '["Cześć!"]', '{"tex', '{"text": 5}'])
async def test_json_reply_without_text_is_an_error(content):
    client = FakeClient(text=content)
    with pytest.raises(EngineError) as e:
        await make(client)[0].respond("x", [])
    assert e.value.kind == "other"
    assert "tts" not in dict(client.calls)


async def test_unknown_gesture_becomes_none():
    reply = await make(FakeClient(text='{"text": "Hej", "gesture": "dance"}'))[0].respond("x", [])
    assert (reply.text, reply.gesture) == ("Hej", "none")


@pytest.mark.parametrize("persona", sorted(p.stem for p in (load_config(env={}).root / "personas").glob("*.md")))
async def test_every_persona_starts_with_the_child_safety_preamble(persona):
    from ohbotchat.engine_chain import SAFETY

    client = FakeClient()
    eng, _ = make(client)
    eng._cfg.conversation.persona = persona
    await eng.respond("x", [])
    system = dict(client.calls)["chat"]["messages"][0]["content"]
    assert system.startswith(SAFETY) and "przemoc" in SAFETY and len(system) > len(SAFETY)


async def test_refused_chat_reply_counts_as_moderated():
    client = FakeClient(text=None)
    orig = client._chat

    async def chat(**kw):
        r = await orig(**kw)
        r.choices[0].message.refusal = "I can't help with that."
        return r

    client.chat.completions.create = chat
    with pytest.raises(EngineError) as e:
        await make(client)[0].respond("x", [])
    assert e.value.kind == "moderated"


async def test_truncated_reply_never_keeps_half_of_a_surrogate_pair():
    reply = await make(FakeClient(text='{"text": "Super \\ud83d\\ude0', finish="length"))[0].respond("x", [])
    assert reply.text == "Super"
    reply.text.encode("utf-8")


def _server_error():
    return openai.InternalServerError("Error code: 500", response=_resp(500), body=None)


@pytest.mark.parametrize("stage", ["chat", "tts", "moderation"])
async def test_server_error_is_retried_once(stage):
    client = FakeClient(fail=(stage, _server_error()))
    go = client._go

    def once(name, kw):
        try:
            go(name, kw)
        finally:
            if name == stage:
                client.fail = None

    client._go = once
    eng, _ = make(client)
    reply = await eng.respond("x", [])
    assert reply.text == "Cześć!"
    assert [n for n, _ in client.calls].count(stage) == (3 if stage == "moderation" else 2)


async def test_server_error_twice_names_the_stage():
    eng, _ = make(FakeClient(fail=("chat", _server_error())))
    with pytest.raises(EngineError) as e:
        await eng.respond("x", [])
    assert e.value.kind == "other" and "chat" in str(e.value)


async def test_say_moderates_then_speaks_without_chat():
    client = FakeClient()
    eng, _ = make(client)
    assert await eng.say("Dzień dobry") == b"\x01\x00" * 10
    assert [n for n, _ in client.calls] == ["moderation", "tts"]
    assert dict(client.calls)["moderation"]["input"] == "Dzień dobry"
    assert dict(client.calls)["tts"]["voice"] == eng._cfg.openai.tts_voice


async def test_say_with_another_voice_and_flagged_text():
    client = FakeClient()
    eng, _ = make(client)
    await eng.say("x", voice="ash")
    assert dict(client.calls)["tts"]["voice"] == "ash"
    client = FakeClient(flagged=(True,))
    eng, _ = make(client)
    with pytest.raises(EngineError) as e:
        await eng.say("brzydkie")
    assert e.value.kind == "moderated" and [n for n, _ in client.calls] == ["moderation"]
