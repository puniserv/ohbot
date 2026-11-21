import asyncio
import threading

from ohbotchat.config import load_config
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
    cfg = load_config(env={})
    assert cfg.openai.max_tokens == 150
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
