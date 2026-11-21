import asyncio
from dataclasses import dataclass, field

QUEUE_SIZE = 200  # a stalled websocket client must not make the queue grow without bound


@dataclass
class Event:
    type: str  # "state" | "stage" | "heard" | "reply" | "error" | "robot" | "level" | "info" | "face" | "camera"
    data: dict = field(default_factory=dict)


class EventBus:
    def __init__(self) -> None:
        self._subs: list[asyncio.Queue[Event]] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """Pins the loop that publish() from other threads targets, even before anyone subscribes."""
        self._loop = loop

    def unsubscribe(self, q: asyncio.Queue[Event]) -> None:
        if q in self._subs:
            self._subs.remove(q)

    def subscribe(self) -> asyncio.Queue[Event]:
        # the first subscriber pins the loop that publish() from other threads targets
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        q: asyncio.Queue[Event] = asyncio.Queue(QUEUE_SIZE)
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
            if q.full():
                q.get_nowait()
            q.put_nowait(event)
