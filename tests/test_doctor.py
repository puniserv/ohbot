from types import SimpleNamespace as NS

import httpx2
import openai

from ohbotchat.config import load_config
from ohbotchat.doctor import run_doctor
from ohbotchat.robot import NullDriver


def _resp(status):
    return httpx2.Response(status, request=httpx2.Request("POST", "http://x"))


class Page:
    def __init__(self, ids):
        self.ids = ids

    def __aiter__(self):
        async def gen():
            for i in self.ids:
                yield NS(id=i)

        return gen()


class FakeClient:
    def __init__(self, models=None, models_error=None, chat_error=None):
        cfg = load_config(env={}).openai
        self.ids = models if models is not None else [cfg.stt_model, cfg.chat_model, cfg.tts_model, cfg.moderation_model]
        self.models_error, self.chat_error, self.chat_calls = models_error, chat_error, []
        self.models = NS(list=self._list)
        self.chat = NS(completions=NS(create=self._chat))

    async def _list(self, **kw):
        self.list_kw = kw
        if self.models_error:
            raise self.models_error
        return Page(self.ids)

    async def _chat(self, **kw):
        self.chat_calls.append(kw)
        if self.chat_error:
            raise self.chat_error
        return NS(choices=[NS(message=NS(content="ok"))])


class FakeSpeaker:
    def __init__(self):
        self.played = 0

    async def play(self, pcm, rate=24000, on_frame=None):
        self.played += 1
        return True


def mic(interactive):
    return ["Test mic"], (0.2 if interactive else None)


def cfg_with_key():
    cfg = load_config(env={})
    cfg.api_key = "sk-test"
    return cfg


async def run(client, cfg=None, interactive=False, driver=None, speaker=None, mic_fn=mic):
    return await run_doctor(
        cfg or cfg_with_key(),
        interactive=interactive,
        client=client,
        driver=driver or NullDriver(),
        speaker=speaker or FakeSpeaker(),
        mic=mic_fn,
    )


async def test_all_good_non_interactive_skips_hardware_actions():
    driver, speaker = NullDriver(), FakeSpeaker()
    res = await run(FakeClient(), driver=driver, speaker=speaker)
    assert len(res) == 8
    assert [r.ok for r in res] == [True, True, True, True, True, True, None, True]
    assert speaker.played == 0 and driver.calls == []


async def test_interactive_runs_speaker_and_robot_motion():
    driver, speaker = NullDriver(), FakeSpeaker()
    res = await run(FakeClient(), interactive=True, driver=driver, speaker=speaker)
    assert all(r.ok for r in res)
    assert speaker.played == 1
    assert any(c[0] == "eye" for c in driver.calls) and any(c[0] == "move" for c in driver.calls)


async def test_missing_key_reports_with_hint_and_no_crash():
    cfg = load_config(env={})
    cfg.api_key = None  # the repo may hold a real .env
    res = await run(None, cfg=cfg, driver=NullDriver(connected=False))
    assert [r.ok for r in res][:5] == [False] * 5
    assert ".env" in res[0].hint
    assert res[7].ok is False and res[7].hint


async def test_quota_detected_by_minimal_chat_request():
    err = openai.RateLimitError(
        "quota", response=_resp(429), body={"code": "insufficient_quota", "message": "x"}
    )
    client = FakeClient(chat_error=err)
    res = await run(client)
    assert res[4].ok is False and "środk" in res[4].detail.lower()
    assert client.chat_calls[0]["max_completion_tokens"] <= 16


async def test_bad_key_fails_key_check():
    err = openai.AuthenticationError("bad", response=_resp(401), body=None)
    res = await run(FakeClient(models_error=err))
    assert res[1].ok is True  # the API answered, so the network is fine
    assert res[2].ok is False and "OPENAI_API_KEY" in res[2].hint


async def test_models_list_uses_the_configured_timeout():
    client, cfg = FakeClient(), cfg_with_key()
    cfg.openai.timeout_s = 3
    await run(client, cfg)
    assert client.list_kw == {"timeout": 3}


async def test_rejected_key_detail_shows_no_key_fragment():
    err = openai.AuthenticationError("Incorrect API key provided: sk-ab***wxyz.", response=_resp(401), body=None)
    res = await run(FakeClient(models_error=err))
    assert "sk-" not in res[2].detail and "odrzucony" in res[2].detail


async def test_network_error_fails_internet_check():
    res = await run(FakeClient(models_error=openai.APIConnectionError(request=httpx2.Request("GET", "http://x"))))
    assert res[1].ok is False
    assert res[4].ok is False


async def test_unknown_model_is_listed():
    res = await run(FakeClient(models=["gpt-5.4-mini"]))
    assert res[3].ok is False
    assert "gpt-transcribe" in res[3].detail


async def test_robot_absent():
    res = await run(FakeClient(), driver=NullDriver(connected=False))
    assert res[7].ok is False and "Ohbot" in res[7].hint or "USB" in res[7].hint


async def test_mic_silence_when_interactive():
    res = await run(FakeClient(), interactive=True, mic_fn=lambda i: (["m"], 0.0))
    assert res[5].ok is False
    res = await run(FakeClient(), interactive=True, mic_fn=lambda i: ([], None))
    assert res[5].ok is False


async def test_check_exception_becomes_failure():
    def boom(interactive):
        raise OSError("no audio backend")

    res = await run(FakeClient(), mic_fn=boom)
    assert res[5].ok is False and "no audio backend" in res[5].detail


async def test_api_error_is_not_reported_as_bad_key():
    err = openai.InternalServerError("boom", response=_resp(500), body=None)
    res = await run(FakeClient(models_error=err))
    assert res[2].ok is False and "odrzucony" not in res[2].detail and "błąd" in res[2].detail
