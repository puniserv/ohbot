from enum import StrEnum


class State(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


_TABLE: dict[tuple[State, str], State] = {
    (State.IDLE, "listen"): State.LISTENING,
    (State.LISTENING, "utterance"): State.TRANSCRIBING,
    (State.LISTENING, "stop"): State.IDLE,
    (State.TRANSCRIBING, "heard"): State.THINKING,
    (State.TRANSCRIBING, "stop"): State.LISTENING,
    (State.THINKING, "reply"): State.SPEAKING,
    (State.THINKING, "stop"): State.LISTENING,
    (State.SPEAKING, "done"): State.IDLE,
    (State.SPEAKING, "stop"): State.LISTENING,
    (State.ERROR, "done"): State.IDLE,
    (State.ERROR, "listen"): State.LISTENING,
}


def transition(state: State, action: str) -> State:
    """Next state for an action; "error" always wins, unknown actions keep the state."""
    if action == "error":
        return State.ERROR
    return _TABLE.get((state, action), state)
