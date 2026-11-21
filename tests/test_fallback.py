from ohbotchat.engine import EngineError
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


def test_get_missing_is_none(tmp_path):
    assert Fallback(tmp_path).get("error") is None


async def test_ensure_generates_missing_only(tmp_path):
    fb, tts = Fallback(tmp_path), FakeTts()
    assert not fb.complete()
    await fb.ensure(tts)
    assert fb.complete() and set(tts.publish_flags) == {False}
    assert set(PHRASES) == {"no_network", "not_heard", "moderated", "error"}
    assert fb.get("not_heard") == PHRASES["not_heard"].encode()
    tts.said.clear()
    await fb.ensure(tts)
    assert tts.said == []


async def test_ensure_survives_offline(tmp_path):
    fb = Fallback(tmp_path)
    await fb.ensure(FakeTts(fail_on=PHRASES["error"]))
    assert fb.get("error") is None and fb.get("moderated") is not None
