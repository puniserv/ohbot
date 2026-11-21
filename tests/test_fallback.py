from ohbotchat.engine import EngineError
from ohbotchat.config import DEFAULT_FILLERS as FILLERS
from ohbotchat.fallback import PHRASES, Fallback


class FakeTts:
    def __init__(self, fail_on=None):
        self.said, self.fail_on, self.publish_flags = [], fail_on, []

    async def synthesize(self, text, publish=True):
        self.publish_flags.append(publish)
        if text == self.fail_on:
            raise EngineError("network")
        self.said.append(text)
        return text.encode()

    async def say(self, text, publish=True):
        if text == "brzydko":
            raise EngineError("moderated")
        self.moderated = [*getattr(self, "moderated", []), text]
        return await self.synthesize(text, publish)


def test_get_missing_is_none(tmp_path):
    assert Fallback(tmp_path).get("error") is None


async def test_ensure_generates_missing_only(tmp_path):
    fb, tts = Fallback(tmp_path), FakeTts()
    assert not fb.complete(FILLERS)
    await fb.ensure(tts, FILLERS)
    assert fb.complete(FILLERS) and set(tts.publish_flags) == {False}
    assert set(PHRASES) == {"no_network", "not_heard", "moderated", "error", "listening"}
    assert fb.get("not_heard") == PHRASES["not_heard"].encode()
    tts.said.clear()
    await fb.ensure(tts, FILLERS)
    assert tts.said == []


async def test_ensure_survives_offline(tmp_path):
    fb = Fallback(tmp_path)
    await fb.ensure(FakeTts(fail_on=PHRASES["error"]), FILLERS)
    assert fb.get("error") is None and fb.get("moderated") is not None


def test_text_cache_is_per_directory_and_text(tmp_path):
    fb = Fallback(tmp_path / "v1")
    assert fb.cached("Cześć") is None
    fb.store("Cześć", b"pcm")
    assert fb.cached("Cześć") == b"pcm" and fb.cached("Hej") is None
    fb.dir = tmp_path / "v2"  # another voice
    assert fb.cached("Cześć") is None


async def test_ensure_also_caches_fillers(tmp_path):
    fb, tts = Fallback(tmp_path), FakeTts()
    await fb.ensure(tts, FILLERS)
    assert all(fb.cached(t) == t.encode() for t in FILLERS)
    fb.dir.joinpath(next(fb.dir.glob("say-*.pcm")).name).unlink()
    assert not fb.complete(FILLERS)


async def test_fillers_are_user_text_so_they_pass_moderation_and_a_flagged_one_is_not_cached(tmp_path):
    fb, tts = Fallback(tmp_path), FakeTts()
    await fb.ensure(tts, ["Moment", "brzydko"])
    assert tts.moderated == ["Moment"] and fb.cached("Moment") == b"Moment" and fb.cached("brzydko") is None
