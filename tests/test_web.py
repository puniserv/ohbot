import asyncio
import re

import pytest
from fastapi.testclient import TestClient

from ohbotchat.config import load_config
from ohbotchat.doctor import CheckResult
from ohbotchat.events import EventBus
from ohbotchat.web import DEFAULT_PORT, create_app, server_config


class FakeRobot:
    connected = True

    def __init__(self):
        self.calls, self.hold_idle = [], False

    def move(self, motor, pos):
        self.calls.append(("move", motor, pos))

    def close_mouth(self):
        self.calls.append("close_mouth")

    def neutral(self):
        self.calls.append("neutral")


class FakeApp:
    def __init__(self):
        self.config = load_config(env={"OPENAI_API_KEY": "sk-test"})
        self.bus = EventBus()
        self.state = "idle"
        self.robot = FakeRobot()
        self.calls = []
        self.listener = type("L", (), {"recording": False})()
        self.vision = type("V", (), {"ok": True})()
        self.closed = 0
        self.mic_error = None

    def set_mode(self, mode):
        self.calls.append(f"mode:{mode}")
        self.config.conversation.mode = mode

    def ptt_abort(self):
        self.calls.append("ptt_abort")

    async def close(self):
        self.closed += 1

    def ptt_start(self):
        self.calls.append("ptt_start")

    def ptt_stop(self):
        self.calls.append("ptt_stop")

    def stop(self):
        self.calls.append("stop")

    def new_conversation(self):
        self.calls.append("new")

    def greet(self):
        self.calls.append("greet")

    def ready(self):
        return self.state in ("idle", "listening")

    def manual_gesture(self, name):
        self.calls.append(f"gesture:{name}")

    def say(self, text):
        self.calls.append(f"say:{text}")

    def voice_sample(self, voice):
        self.calls.append(f"sample:{voice}")

    def apply_settings(self, values):
        self.calls.append(("apply", values))
        for section, keys in values.items():
            for k, v in keys.items():
                setattr(getattr(self.config, section), k, v)


async def fake_doctor(app):
    return [CheckResult("Klucz", True, "jest"), CheckResult("Mikrofon", False, "brak", "podłącz")]


@pytest.fixture
def app():
    return FakeApp()


@pytest.fixture
def client(app):
    return TestClient(create_app(app, doctor=fake_doctor, allowed_hosts={"testserver"}))


ORIGIN = {"origin": "http://testserver"}


def connect(client, headers=ORIGIN):
    return client.websocket_connect("/ws", headers=headers)


def snapshot(ws):
    msg = ws.receive_json()
    assert msg["type"] == "snapshot"
    return msg["data"]


def sync(ws, app):
    """Round-trips an event so earlier commands are processed."""
    app.bus.publish("info", text="sync")
    while (msg := ws.receive_json())["type"] != "info":
        pass


def test_index_and_static_are_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Ohbot" in r.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200


def test_snapshot_contents(client, app):
    with connect(client) as ws:
        s = snapshot(ws)
    assert s["state"] == "idle" and s["mode"] == "ptt" and s["persona"] == "dzieci"
    assert {"dzieci", "pokaz", "domowy"} <= set(s["personas"])
    assert s["robot"] is True and s["has_key"] is True
    assert s["barge_in"] == {"enabled": False, "threshold": app.config.barge_in.threshold}
    assert s["colours"]["listening"] == app.config.colours.listening
    assert s["mic_error"] is None


def test_snapshot_carries_a_mic_error_raised_before_the_client_connected(client, app):
    app.mic_error = "Nie mogę włączyć mikrofonu"
    with connect(client) as ws:
        assert snapshot(ws)["mic_error"] == "Nie mogę włączyć mikrofonu"


def test_missing_key_is_reported_in_snapshot(app):
    app.config.api_key = None
    with connect(TestClient(create_app(app, doctor=fake_doctor, allowed_hosts={"testserver"}))) as ws:
        assert snapshot(ws)["has_key"] is False


def test_bus_events_are_streamed(client, app):
    with connect(client) as ws:
        snapshot(ws)
        app.bus.publish("heard", text="Cześć")
        assert ws.receive_json() == {"type": "heard", "data": {"text": "Cześć"}}


@pytest.mark.parametrize(
    "cmd,call",
    [("ptt_start", ["ptt_start"]), ("ptt_stop", ["ptt_stop"]), ("stop", ["stop"]), ("new", ["new", "greet"])],
)
def test_simple_commands_call_app(client, app, cmd, call):
    with connect(client) as ws:
        snapshot(ws)
        ws.send_json({"cmd": cmd})
        sync(ws, app)
    assert app.calls == call


def test_mode_persona_and_barge_in_are_stored_and_broadcast(client, app):
    with connect(client) as ws:
        snapshot(ws)
        ws.send_json({"cmd": "mode", "value": "wake"})
        assert ws.receive_json()["data"]["mode"] == "wake"
        ws.send_json({"cmd": "persona", "value": "pokaz"})
        assert ws.receive_json()["data"]["persona"] == "pokaz"
        ws.send_json({"cmd": "barge_in", "enabled": True, "threshold": 0.4})
        assert ws.receive_json()["data"]["barge_in"] == {"enabled": True, "threshold": 0.4}
    assert (app.config.conversation.mode, app.config.conversation.persona) == ("wake", "pokaz")
    assert app.config.barge_in.enabled and app.config.barge_in.threshold == 0.4
    assert app.calls == ["mode:wake", "new"]  # persona switch starts a fresh conversation


@pytest.mark.parametrize(
    "msg",
    [
        {"cmd": "mode", "value": "bogus"},
        {"cmd": "persona", "value": "../../etc/passwd"},
        {"cmd": "barge_in", "enabled": "x", "threshold": "y"},
        {"cmd": "barge_in", "enabled": True, "threshold": 7},
        {"cmd": "nope"},
        {"no_cmd": 1},
        [1, 2],
        "str",
        None,
    ],
)
def test_invalid_commands_are_ignored(client, app, msg):
    with connect(client) as ws:
        snapshot(ws)
        ws.send_json(msg)
        sync(ws, app)
    assert app.config.conversation.mode == "ptt" and app.config.conversation.persona == "dzieci"
    assert not app.config.barge_in.enabled and app.config.barge_in.threshold == 0.15
    assert app.calls == []


def test_malformed_json_keeps_socket_open(client, app):
    with connect(client) as ws:
        snapshot(ws)
        ws.send_text("{not json")
        ws.send_json({"cmd": "stop"})
        sync(ws, app)
    assert app.calls == ["stop"]


def test_doctor_results_are_published_as_event(client, app):
    with connect(client) as ws:
        snapshot(ws)
        ws.send_json({"cmd": "doctor"})
        msg = ws.receive_json()
    assert msg["type"] == "doctor"
    assert msg["data"]["results"] == [
        {"name": "Klucz", "ok": True, "detail": "jest", "hint": ""},
        {"name": "Mikrofon", "ok": False, "detail": "brak", "hint": "podłącz"},
    ]


def test_disconnect_unsubscribes(client, app):
    with connect(client) as ws:
        snapshot(ws)
        assert len(app.bus._subs) == 1
    client.get("/")  # lets the server side finish the handler
    assert app.bus._subs == []


def test_server_binds_localhost_only_on_given_port(app):
    c = server_config(create_app(app), 8799)
    assert (c.host, c.port) == ("127.0.0.1", 8799)
    assert server_config(create_app(app)).port == DEFAULT_PORT == 8765


def test_foreign_missing_and_rebinding_origins_are_rejected(client):
    from starlette.websockets import WebSocketDisconnect

    for headers in ({"origin": "http://evil.example"}, {}, {"origin": "http://testserver:9999"}, {"origin": "null"}):
        with pytest.raises(WebSocketDisconnect) as e, connect(client, headers):
            pass
        assert e.value.code == 1008


def test_unlisted_host_is_rejected_even_with_matching_origin(app):
    from starlette.websockets import WebSocketDisconnect

    c = TestClient(create_app(app, doctor=fake_doctor))  # default hosts: localhost only
    with pytest.raises(WebSocketDisconnect), connect(c):
        pass


def test_same_origin_is_accepted(client):
    with connect(client) as ws:
        assert ws.receive_json()["type"] == "snapshot"


def test_shutdown_closes_the_app(app):
    with TestClient(create_app(app, doctor=fake_doctor, allowed_hosts={"testserver"})):
        assert app.closed == 0
    assert app.closed == 1


def test_disconnect_while_recording_discards_audio(client, app):
    app.listener.recording = True
    with connect(client) as ws:
        snapshot(ws)
    client.get("/")
    assert app.calls == ["ptt_abort"]


def test_doctor_is_ignored_while_a_run_is_pending(app):
    started = []

    async def slow(app):
        started.append(1)
        await asyncio.sleep(0.3)
        return []

    c = TestClient(create_app(app, doctor=slow, allowed_hosts={"testserver"}))
    with connect(c) as ws:
        snapshot(ws)
        ws.send_json({"cmd": "doctor"})
        ws.send_json({"cmd": "doctor"})
        assert ws.receive_json()["type"] == "doctor"
    assert started == [1]


async def test_web_doctor_never_touches_the_robot_driver(monkeypatch):
    from ohbotchat import doctor, web

    touched = []

    class Spy:
        def __init__(self):
            touched.append("created")

        def connect(self):
            touched.append("connect")
            return True

        def close(self):
            touched.append("close")

    class Quiet:
        async def play(self, pcm, *a, **k):
            return True

    monkeypatch.setattr(doctor, "OhbotDriver", Spy)
    monkeypatch.setattr("ohbotchat.speak.Speaker", Quiet)
    app = FakeApp()
    app.config.api_key = None
    results = await web._run_doctor(app)
    assert touched == [] and next(r for r in results if r.name == "Ohbot").ok is True
    assert next(r for r in results if r.name == "Kamera").ok is True  # from the running app, not a second open


@pytest.mark.parametrize(
    "msg,call",
    [
        ({"cmd": "gesture", "name": "yes"}, "gesture:yes"),
        ({"cmd": "say", "text": " Dzień dobry "}, "say:Dzień dobry"),
        ({"cmd": "voice_sample", "voice": "ash"}, "sample:ash"),
    ],
)
def test_test_commands_call_app_when_idle(client, app, msg, call):
    with connect(client) as ws:
        snapshot(ws)
        ws.send_json(msg)
        sync(ws, app)
    assert app.calls == [call]


def test_motor_commands_are_clamped_and_reach_the_robot(client, app):
    with connect(client) as ws:
        snapshot(ws)
        ws.send_json({"cmd": "motor", "motor": 4, "pos": 12})
        ws.send_json({"cmd": "motor", "motor": 0, "pos": 2.5})
        ws.send_json({"cmd": "close_mouth"})
        ws.send_json({"cmd": "neutral"})
        sync(ws, app)
    assert app.robot.calls == [("move", 4, 10), ("move", 0, 2.5), "close_mouth", "neutral"]


@pytest.mark.parametrize(
    "msg",
    [
        {"cmd": "gesture", "name": "none"},
        {"cmd": "gesture", "name": "dance"},
        {"cmd": "motor", "motor": 7, "pos": 5},
        {"cmd": "motor", "motor": True, "pos": 5},
        {"cmd": "motor", "motor": 1, "pos": "5"},
        {"cmd": "say", "text": ""},
        {"cmd": "say", "text": "x" * 301},
        {"cmd": "voice_sample", "voice": "darth"},
        {"cmd": "test_panel", "open": "yes"},
    ],
)
def test_invalid_test_commands_are_ignored(client, app, msg):
    with connect(client) as ws:
        snapshot(ws)
        ws.send_json(msg)
        sync(ws, app)
    assert app.calls == [] and app.robot.calls == [] and app.robot.hold_idle is False


@pytest.mark.parametrize(
    "msg",
    [{"cmd": "gesture", "name": "yes"}, {"cmd": "motor", "motor": 1, "pos": 5}, {"cmd": "close_mouth"}, {"cmd": "neutral"}, {"cmd": "say", "text": "a"}, {"cmd": "voice_sample", "voice": "ash"}],
)
def test_test_commands_are_rejected_unless_idle(client, app, msg):
    app.state = "speaking"
    with connect(client) as ws:
        snapshot(ws)
        ws.send_json(msg)
        e = ws.receive_json()
    assert e["type"] == "error" and "zajęty" in e["data"]["message"]
    assert app.calls == [] and app.robot.calls == []


def test_test_panel_holds_idle_moves_until_closed_or_disconnect(client, app):
    with connect(client) as ws:
        snapshot(ws)
        ws.send_json({"cmd": "test_panel", "open": True})
        sync(ws, app)
        assert app.robot.hold_idle is True
        ws.send_json({"cmd": "test_panel", "open": False})
        sync(ws, app)
        assert app.robot.hold_idle is False
        ws.send_json({"cmd": "test_panel", "open": True})
        sync(ws, app)
    client.get("/")
    assert app.robot.hold_idle is False


def test_get_settings(client, app):
    r = client.get("/api/settings")
    assert r.status_code == 200
    d = r.json()
    assert d["values"]["robot"]["lip_closed"] == app.config.robot.lip_closed
    assert "cedar" in d["voices"] and "none" not in d["gestures"] and "yes" in d["gestures"]


def test_post_settings_applies_live_and_broadcasts(client, app):
    with connect(client) as ws:
        snapshot(ws)
        r = client.post("/api/settings", json={"barge_in": {"threshold": 0.3}, "robot": {"lip_open": 7}}, headers=ORIGIN)
        assert r.status_code == 200
        assert ws.receive_json()["data"]["barge_in"]["threshold"] == 0.3
    assert app.calls == [("apply", {"barge_in": {"threshold": 0.3}, "robot": {"lip_open": 7.0}})]


@pytest.mark.parametrize(
    "body", [{"robot": {"nope": 1}}, {"robot": {"lip_open": 99}}, {"openai": {"tts_voice": "darth"}}, [1]]
)
def test_post_settings_rejects_unknown_or_invalid(client, app, body):
    r = client.post("/api/settings", json=body, headers=ORIGIN)
    assert r.status_code == 400 and app.calls == []


def test_post_settings_rejects_malformed_json(client, app):
    r = client.post("/api/settings", content="{x", headers={**ORIGIN, "content-type": "application/json"})
    assert r.status_code == 400


@pytest.mark.parametrize("headers", [{}, {"origin": "http://evil.example"}])
def test_api_writes_need_same_origin(client, app, headers):
    assert client.post("/api/settings", json={"robot": {"lip_open": 7}}, headers=headers).status_code == 403
    assert client.post("/api/settings/save", headers=headers).status_code == 403
    assert client.put("/api/persona/dzieci", json={"text": "x"}, headers=headers).status_code == 403
    assert app.calls == []


def test_api_rejects_unlisted_host(app):
    c = TestClient(create_app(app, doctor=fake_doctor))
    assert c.get("/api/settings").status_code == 403


def test_save_writes_config_toml_keeping_comments(client, app, tmp_path):
    app.config.root = tmp_path
    (tmp_path / "config.toml").write_text("[robot]\n# measured\nlip_closed = 1.5\n", encoding="utf-8")
    app.config.robot.lip_closed = 2.0
    assert client.post("/api/settings/save", headers=ORIGIN).status_code == 200
    text = (tmp_path / "config.toml").read_text(encoding="utf-8")
    assert "# measured\nlip_closed = 2.0\n" in text and "gesture_scale = 1.0" in text


def test_persona_text_read_and_written(client, app, tmp_path):
    import shutil

    shutil.copytree(app.config.root / "personas", tmp_path / "personas")
    app.config.root = tmp_path
    r = client.get("/api/persona/pokaz")
    assert r.status_code == 200 and r.json()["text"] == (tmp_path / "personas" / "pokaz.md").read_text(encoding="utf-8")
    assert client.put("/api/persona/pokaz", json={"text": "Nowy prompt"}, headers=ORIGIN).status_code == 200
    assert (tmp_path / "personas" / "pokaz.md").read_text(encoding="utf-8") == "Nowy prompt"
    assert client.put("/api/persona/pokaz", json={"text": "x" * 8001}, headers=ORIGIN).status_code == 400
    assert client.put("/api/persona/pokaz", json={"text": 5}, headers=ORIGIN).status_code == 400
    assert client.put("/api/persona/nowa", json={"text": "x"}, headers=ORIGIN).status_code == 404
    assert client.get("/api/persona/..%2Fconfig").status_code == 404
    assert not (tmp_path / "personas" / "nowa.md").exists()


def test_idle_is_held_while_any_tab_has_the_test_panel_open(client, app):
    with connect(client) as a:
        snapshot(a)
        with connect(client) as b:
            snapshot(b)
            a.send_json({"cmd": "test_panel", "open": True})
            b.send_json({"cmd": "test_panel", "open": True})
            sync(a, app)
            sync(b, app)
            a.send_json({"cmd": "test_panel", "open": False})
            sync(a, app)
            assert app.robot.hold_idle is True  # tab b still has it open
            b.send_json({"cmd": "test_panel", "open": False})
            sync(b, app)
            assert app.robot.hold_idle is False
            b.send_json({"cmd": "test_panel", "open": True})
            sync(b, app)
        client.get("/")
        assert app.robot.hold_idle is False  # b went away with its panel open
        a.send_json({"cmd": "test_panel", "open": False})  # a never reopened; closing twice must not go negative
        sync(a, app)
        assert app.robot.hold_idle is False


def test_snapshot_reports_the_camera(client, app):
    app.vision.ok = False
    with connect(client) as ws:
        assert snapshot(ws)["camera"] is False


def test_camera_settings_are_served_and_applied(client, app):
    assert client.get("/api/settings").json()["values"]["camera"]["index"] == 0
    r = client.post("/api/settings", json={"camera": {"index": 1, "mirror": True}}, headers=ORIGIN)
    assert r.status_code == 200 and app.config.camera.index == 1 and app.config.camera.mirror is True
    assert client.post("/api/settings", json={"camera": {"index": 10}}, headers=ORIGIN).status_code == 400


def test_every_settings_field_in_the_page_is_an_editable_key(client):
    from ohbotchat.config import EDITABLE

    keys = re.findall(r'data-key="(\w+)\.(\w+)"', client.get("/").text)
    assert all(k in EDITABLE[s] for s, k in keys)
    assert {k for s, k in keys if s == "camera"} == set(EDITABLE["camera"])


async def test_web_doctor_reports_a_camera_that_is_still_starting():
    from ohbotchat import web

    app = FakeApp()
    app.config.api_key = None
    app.vision.ok = None
    kamera = next(r for r in await web._run_doctor(app) if r.name == "Kamera")
    assert kamera.ok is None and kamera.detail == "kamera uruchamia się"
