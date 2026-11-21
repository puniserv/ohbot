import asyncio
import threading

import pytest

from ohbotchat.config import Config, load_config
from ohbotchat.events import EventBus
from ohbotchat.state import State, transition


def test_config_defaults_and_env_key(tmp_path):
    (tmp_path / "config.toml").write_text('[conversation]\npersona = "pokaz"\n')
    cfg = load_config(tmp_path, env={"OPENAI_API_KEY": "sk-test"})
    assert cfg.api_key == "sk-test"
    assert cfg.conversation.persona == "pokaz"
    assert cfg.conversation.history_turns == 10


def test_missing_key_is_none(tmp_path):
    assert load_config(tmp_path, env={}).api_key is None


def test_dotenv_key(tmp_path):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-file\n")
    assert load_config(tmp_path, env={}).api_key == "sk-file"


def test_repo_config_toml_loads():
    cfg = Config()
    assert cfg.openai.max_tokens == 200
    assert cfg.colours.listening == [0, 10, 0]
    assert "ohbot" in cfg.wake.words


async def test_bus_multiple_subscribers():
    bus = EventBus()
    a, b = bus.subscribe(), bus.subscribe()
    bus.publish("info", msg="x")
    assert (await a.get()).data == {"msg": "x"}
    assert (await b.get()).type == "info"


async def test_bus_from_foreign_thread():
    bus = EventBus()
    q = bus.subscribe()
    t = threading.Thread(target=lambda: bus.publish("level", v=1))
    t.start()
    ev = await asyncio.wait_for(q.get(), 1)
    t.join()
    assert ev.type == "level" and ev.data["v"] == 1


def test_state_transitions():
    assert transition(State.SPEAKING, "stop") == State.LISTENING
    assert transition(State.IDLE, "listen") == State.LISTENING
    assert transition(State.THINKING, "error") == State.ERROR
    assert transition(State.IDLE, "bogus") == State.IDLE


async def test_bus_unsubscribe():
    bus = EventBus()
    a, b = bus.subscribe(), bus.subscribe()
    bus.unsubscribe(a)
    bus.publish("info")
    assert a.empty() and not b.empty()


async def test_bus_bound_loop_is_safe_before_first_subscribe():
    bus = EventBus()
    bus.bind(asyncio.get_running_loop())
    t = threading.Thread(target=lambda: bus.publish("info", n=0))
    t.start()
    t.join()
    q = bus.subscribe()
    t = threading.Thread(target=lambda: bus.publish("info", n=1))
    t.start()
    got = {(await asyncio.wait_for(q.get(), 1)).data["n"] for _ in range(2)}  # n=0 was queued before subscribe
    t.join()
    assert got == {0, 1}


def test_cancel_goes_idle_from_busy_states():
    for st in (State.TRANSCRIBING, State.THINKING, State.SPEAKING, State.LISTENING, State.ERROR):
        assert transition(st, "cancel") == State.IDLE


async def test_bus_full_queue_drops_oldest_instead_of_growing():
    from ohbotchat.events import QUEUE_SIZE

    bus = EventBus()
    q = bus.subscribe()
    for i in range(QUEUE_SIZE + 5):
        bus.publish("level", n=i)
    assert q.qsize() == QUEUE_SIZE
    assert q.get_nowait().data["n"] == 5 and q._queue[-1].data["n"] == QUEUE_SIZE + 4


def test_app_root_is_the_exe_folder_in_a_frozen_build(monkeypatch, tmp_path):
    import sys

    from ohbotchat.config import ROOT, app_root

    assert app_root() == ROOT
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "ohbotchat.exe"))
    assert app_root() == tmp_path.resolve()


def test_validate_settings_accepts_known_keys_and_normalises():
    from ohbotchat.config import validate_settings

    out = validate_settings({"robot": {"lip_closed": 2, "lip_invert_top": True}, "openai": {"tts_voice": "ash"}})
    assert out == {"robot": {"lip_closed": 2.0, "lip_invert_top": True}, "openai": {"tts_voice": "ash"}}
    assert isinstance(out["robot"]["lip_closed"], float)


def test_camera_settings_are_editable_and_normalised():
    from ohbotchat.config import editable_settings, validate_settings

    values = editable_settings(Config())["camera"]
    assert values == {
        "enabled": True, "index": 0, "mirror": False, "offset_x": 0.0, "offset_y": 0.0, "head_gain": 3.0,
        "eye_gain": 10.0, "deadband": 0.5, "smoothing": 0.5, "min_score": 0.7, "lost_after_s": 3.0,
        "auto_greet": True, "greet_cooldown_s": 60.0,
    }
    out = validate_settings({"camera": {"index": 2, "mirror": True, "offset_x": -1, "greet_cooldown_s": 120}})
    assert out == {"camera": {"index": 2, "mirror": True, "offset_x": -1.0, "greet_cooldown_s": 120.0}}


@pytest.mark.parametrize(
    "bad",
    [
        {"robot": {"nope": 1}},
        {"nope": {"x": 1}},
        {"openai": {"chat_model": "gpt-x"}},  # exists, but not editable
        {"robot": {"lip_closed": 11}},
        {"robot": {"lip_closed": "2"}},
        {"robot": {"lip_closed": True}},
        {"robot": {"gesture_speed": 2.5}},
        {"robot": {"gesture_scale": 1.6}},
        {"robot": {"lip_invert_top": 1}},
        {"openai": {"tts_voice": "darth"}},
        {"conversation": {"greeting": ""}},
        {"conversation": {"greeting": "x" * 301}},
        {"conversation": {"greeting": "Cześć\x7f"}},  # DEL written raw makes tomllib reject the file
        {"conversation": {"greeting": "Cześć\nhej"}},
        {"conversation": {"greeting": "a\x00b"}},
        {"camera": {"index": 10}},
        {"camera": {"index": 1.5}},
        {"camera": {"offset_x": 1.1}},
        {"camera": {"smoothing": 0.96}},
        {"camera": {"min_score": 0.2}},
        {"camera": {"lost_after_s": 0.4}},
        {"camera": {"greet_cooldown_s": 5}},
        {"camera": {"enabled": "yes"}},
        {"robot": [1]},
        [1],
    ],
)
def test_validate_settings_rejects(bad):
    from ohbotchat.config import validate_settings

    with pytest.raises(ValueError):
        validate_settings(bad)


def test_save_settings_replaces_only_known_lines_and_keeps_comments(tmp_path):
    from ohbotchat.config import save_settings

    path = tmp_path / "config.toml"
    path.write_text(
        '# top\n[openai]\n# voice\ntts_voice = "cedar"  # male # really\nchat_model = "x"\n\n'
        "[robot]\nlip_closed = 1.5\n# trailing\n\n[vad]\nsilence_ms = 800\n",
        encoding="utf-8",
    )
    save_settings(
        path,
        {
            "openai": {"tts_voice": 'a"b#c'},
            "robot": {"lip_closed": 2.0, "lip_invert_top": True},
            "vad": {"silence_ms": 900},
            "wake": {"window_s": 5.0},
        },
    )
    text = path.read_text(encoding="utf-8")
    assert text == (
        '# top\n[openai]\n# voice\ntts_voice = "a\\"b#c"  # male # really\nchat_model = "x"\n\n'
        "[robot]\nlip_closed = 2.0\n# trailing\nlip_invert_top = true\n\n[vad]\nsilence_ms = 900\n"
        "\n[wake]\nwindow_s = 5.0\n"
    )
    import tomllib

    assert tomllib.loads(text)["openai"]["tts_voice"] == 'a"b#c'
    cfg = load_config(tmp_path, env={})
    assert cfg.robot.lip_closed == 2.0 and cfg.wake.window_s == 5.0 and cfg.robot.lip_invert_top is True


def test_editable_settings_reads_current_values():
    from ohbotchat.config import EDITABLE, editable_settings

    cfg = Config()
    values = editable_settings(cfg)
    assert set(values) == set(EDITABLE) and values["conversation"]["greeting"] == cfg.conversation.greeting
    assert values["conversation"]["fillers"] is True


def test_save_settings_replaces_the_file_atomically(tmp_path, monkeypatch):
    import os

    from ohbotchat import config as config_mod
    from ohbotchat.config import save_settings

    path = tmp_path / "config.toml"
    path.write_text("[robot]\nlip_closed = 1.5\n", encoding="utf-8")
    path.chmod(0o640)
    replaced = []
    real = os.replace
    monkeypatch.setattr(config_mod.os, "replace", lambda a, b: (replaced.append((a, b)), real(a, b)))
    save_settings(path, {"robot": {"lip_closed": 2.0}})
    assert replaced and replaced[0][1] == path and "lip_closed = 2.0" in path.read_text(encoding="utf-8")
    assert [p.name for p in tmp_path.iterdir()] == ["config.toml"]  # no temp file left behind
    assert path.stat().st_mode & 0o777 == 0o640


def test_write_atomic_leaves_the_old_file_when_writing_fails(tmp_path, monkeypatch):
    from ohbotchat import config as config_mod

    path = tmp_path / "x.md"
    path.write_text("old", encoding="utf-8")
    monkeypatch.setattr(config_mod.os, "replace", lambda a, b: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        config_mod.write_atomic(path, "new")
    assert path.read_text(encoding="utf-8") == "old" and [p.name for p in tmp_path.iterdir()] == ["x.md"]


def test_config_toml_lists_every_camera_key():
    import tomllib
    from dataclasses import fields

    from ohbotchat.config import ROOT, CameraConfig

    raw = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
    # load_config skips unknown keys silently, so a typo here would quietly fall back to the default
    assert set(raw["camera"]) == {f.name for f in fields(CameraConfig)}


def test_filler_phrases_are_an_editable_list_of_short_texts():
    from ohbotchat.config import editable_settings, validate_settings

    assert editable_settings(Config())["conversation"]["filler_phrases"] == ["Hmm…", "Chwileczkę…", "Niech pomyślę…", "Już mówię…"]
    out = validate_settings({"conversation": {"filler_phrases": [" Hmm… ", "Moment"]}})
    assert out == {"conversation": {"filler_phrases": ["Hmm…", "Moment"]}}


@pytest.mark.parametrize(
    "bad", [[], ["a"] * 11, ["x" * 41], ["a\x7f"], [""], "Hmm", [1], None],
)
def test_filler_phrases_reject_bad_lists(bad):
    from ohbotchat.config import validate_settings

    with pytest.raises(ValueError):
        validate_settings({"conversation": {"filler_phrases": bad}})


def test_save_settings_writes_a_toml_array_and_replaces_a_multiline_one(tmp_path):
    import tomllib

    from ohbotchat.config import save_settings

    path = tmp_path / "config.toml"
    path.write_text('[conversation]\n# fillers\nfiller_phrases = [\n  "Hmm…",\n  "Hej",\n]  # keep\nfillers = true\n', encoding="utf-8")
    save_settings(path, {"conversation": {"filler_phrases": ['Mo"ment', "Już…"]}})
    text = path.read_text(encoding="utf-8")
    assert text == '[conversation]\n# fillers\nfiller_phrases = ["Mo\\"ment", "Już…"]  # keep\nfillers = true\n'
    assert tomllib.loads(text)["conversation"]["filler_phrases"] == ['Mo"ment', "Już…"]


def test_save_settings_keeps_the_rest_of_the_file_after_a_broken_bracket_line(tmp_path):
    from ohbotchat.config import save_settings

    path = tmp_path / "config.toml"
    path.write_text('[conversation]\nfiller_phrases = ["Hmm…"  # typo\nfillers = true\n\n[robot]\nlip_closed = 1.5\n', encoding="utf-8")
    save_settings(path, {"conversation": {"filler_phrases": ["Ok"]}})
    assert path.read_text(encoding="utf-8") == '[conversation]\nfiller_phrases = ["Ok"]\nfillers = true\n\n[robot]\nlip_closed = 1.5\n'
