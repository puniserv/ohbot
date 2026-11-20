import asyncio
from dataclasses import dataclass, field


@dataclass
class Event:
    type: str  # "state" | "stage" | "heard" | "reply" | "error" | "robot" | "level" | "info"
    data: dict = field(default_factory=dict)


class EventBus:
    def __init__(self) -> None:
        self._subs: list[asyncio.Queue[Event]] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def subscribe(self) -> asyncio.Queue[Event]:
        # the first subscriber pins the loop that publish() from other threads targets
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        q: asyncio.Queue[Event] = asyncio.Queue()
        self._subs.append(q)
        return q

    def publish(self, type: str, **data) -> None:
        event = Event(type, data)
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if self._loop is None or running is self._loop:
            self._deliver(event)
        else:
            self._loop.call_soon_threadsafe(self._deliver, event)

    def _deliver(self, event: Event) -> None:
        for q in self._subs:
            q.put_nowait(event)
