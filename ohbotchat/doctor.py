import asyncio
import math
import struct
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .audio import level
from .config import Config
from .engine import EngineError
from .engine_chain import _mapped_errors
from .robot import HEADNOD, OhbotDriver

SKIPPED = "pominięto"
NO_KEY = "brak klucza API"


@dataclass
class CheckResult:
    name: str
    ok: bool | None  # None = skipped
    detail: str
    hint: str = ""


def _default_mic(interactive: bool) -> tuple[list[str], float | None]:
    import numpy as np
    import sounddevice as sd

    names = [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
    if not interactive or not names:
        return names, None
    rec = sd.rec(2 * 16000, samplerate=16000, channels=1, dtype="int16")
    sd.wait()
    return names, level(np.asarray(rec).tobytes())


def _tone(rate: int = 24000, seconds: float = 0.6, hz: int = 440) -> bytes:
    n = int(rate * seconds)
    return struct.pack(f"<{n}h", *(int(8000 * math.sin(2 * math.pi * hz * i / rate)) for i in range(n)))


def _robot_motion(driver) -> None:
    driver.eye_colour(0, 10, 0)
    driver.move(HEADNOD, 7, 5)
    time.sleep(0.5)
    driver.move(HEADNOD, 5, 5)
    time.sleep(0.5)
    driver.eye_colour(0, 0, 3)


async def run_doctor(
    config: Config,
    interactive: bool = False,
    client=None,
    driver=None,
    speaker=None,
    mic: Callable[[bool], tuple[list[str], float | None]] = _default_mic,
) -> list[CheckResult]:
    if client is None and config.api_key:
        import openai

        client = openai.AsyncOpenAI(api_key=config.api_key, max_retries=0)
    c = config.openai
    api: dict = {}  # shared result of the one models.list() call behind checks 2-4

    def key_check() -> CheckResult:
        name = "Klucz API w .env"
        if config.api_key:
            return CheckResult(name, True, "OPENAI_API_KEY ustawiony")
        return CheckResult(name, False, "brak OPENAI_API_KEY", "Skopiuj .env.example do .env i wpisz klucz z platform.openai.com.")

    async def fetch_models() -> None:
        start = time.perf_counter()
        try:
            with _mapped_errors():
                page = await client.models.list()
                api["ids"] = {m.id async for m in page}
        except EngineError as e:
            api["error"] = e
        except Exception as e:
            api["error"] = EngineError("other", str(e))
        api["latency"] = time.perf_counter() - start

    def internet() -> CheckResult:
        name = "Internet i API"
        e = api.get("error")
        if e and e.kind == "network":
            return CheckResult(name, False, "brak połączenia z api.openai.com", "Sprawdź Wi-Fi, VPN i proxy, potem uruchom ponownie.")
        return CheckResult(name, True, f"API odpowiada w {api['latency'] * 1000:.0f} ms")

    def key_valid() -> CheckResult:
        name = "Klucz ważny"
        e = api.get("error")
        if e is None:
            return CheckResult(name, True, "klucz przyjęty")
        if e.kind == "network":
            return CheckResult(name, False, f"{SKIPPED}: brak połączenia", "Najpierw napraw połączenie z internetem.")
        if e.kind == "auth":
            return CheckResult(name, False, f"klucz odrzucony ({e})", "Sprawdź OPENAI_API_KEY w .env; klucz mógł wygasnąć albo zostać odwołany.")
        return CheckResult(name, False, f"API zwróciło błąd ({e})", "Spróbuj ponownie za chwilę; jeśli błąd wraca, sprawdź status.openai.com.")

    def models() -> CheckResult:
        name = "Modele z config.toml"
        if "error" in api:
            return CheckResult(name, False, f"{SKIPPED}: lista modeli niedostępna", "Najpierw napraw poprzednie kroki.")
        wanted = [c.stt_model, c.chat_model, c.tts_model, c.moderation_model]
        missing = [m for m in wanted if m not in api["ids"]]
        if missing:
            return CheckResult(name, False, "niedostępne: " + ", ".join(missing), "Zmień nazwy modeli w config.toml na dostępne dla tego klucza.")
        return CheckResult(name, True, "wszystkie dostępne")

    async def quota() -> CheckResult:
        name = "Środki na koncie"
        if "error" in api:
            return CheckResult(name, False, f"{SKIPPED}: klucz lub połączenie nie działa", "Najpierw napraw poprzednie kroki.")
        try:
            with _mapped_errors():
                await client.chat.completions.create(
                    model=c.chat_model,
                    messages=[{"role": "user", "content": "ok"}],
                    max_completion_tokens=16,
                    timeout=c.timeout_s,
                    **({"reasoning_effort": c.reasoning_effort} if c.reasoning_effort else {}),
                )
        except EngineError as e:
            if e.kind == "quota":
                return CheckResult(name, False, "brak środków na koncie OpenAI", "Doładuj konto w platform.openai.com, sekcja Billing.")
            return CheckResult(name, False, f"zapytanie testowe nie przeszło ({e})", "Sprawdź model czatu w config.toml.")
        return CheckResult(name, True, "zapytanie testowe przeszło")

    async def mic_check() -> CheckResult:
        name = "Mikrofon"
        names, lvl = await asyncio.to_thread(mic, interactive)
        if not names:
            return CheckResult(name, False, "nie znaleziono mikrofonu", "Podłącz mikrofon i nadaj aplikacji dostęp do niego w ustawieniach systemu.")
        listing = ", ".join(names)
        if lvl is None:
            return CheckResult(name, True, f"urządzenia: {listing}")
        if lvl < 0.005:
            return CheckResult(name, False, f"cisza w nagraniu ({listing})", "Mikrofon wyciszony albo bez uprawnień; sprawdź ustawienia systemu.")
        return CheckResult(name, True, f"poziom {lvl:.2f} ({listing})")

    async def speaker_check() -> CheckResult:
        name = "Głośnik"
        if not interactive:
            return CheckResult(name, None, f"{SKIPPED} (tryb nieinteraktywny)")
        sp = speaker
        if sp is None:
            from .speak import Speaker

            sp = Speaker()
        await sp.play(_tone())
        return CheckResult(name, True, "dźwięk testowy odtworzony; czy go słychać?", "Jeśli nie, sprawdź głośność i wyjście audio w systemie.")

    async def robot_check() -> CheckResult:
        name = "Ohbot"
        d = driver or OhbotDriver()
        try:
            ok = await asyncio.to_thread(d.connect)
            if not ok:
                return CheckResult(name, False, "robot nie odpowiada", "Podłącz Ohbota kablem USB, włącz zasilanie i uruchom ponownie.")
            port = getattr(d, "port", "")
            if interactive:
                await asyncio.to_thread(_robot_motion, d)
            return CheckResult(name, True, f"połączony {port}".strip() + ("" if interactive else "; ruchy pominięte"))
        finally:
            if driver is None:
                await asyncio.to_thread(d.close)

    async def guarded(name: str, fn: Callable[[], Awaitable[CheckResult] | CheckResult]) -> CheckResult:
        try:
            r = fn()
            return await r if asyncio.iscoroutine(r) else r
        except Exception as e:
            return CheckResult(name, False, str(e) or type(e).__name__, "Nieoczekiwany błąd; szczegóły w komunikacie.")

    results = [key_check()]
    if client is None:
        for name in ("Internet i API", "Klucz ważny", "Modele z config.toml", "Środki na koncie"):
            results.append(CheckResult(name, False, f"{SKIPPED}: {NO_KEY}", "Najpierw uzupełnij .env."))
    else:
        await fetch_models()
        results += [
            await guarded("Internet i API", internet),
            await guarded("Klucz ważny", key_valid),
            await guarded("Modele z config.toml", models),
            await guarded("Środki na koncie", quota),
        ]
    results += [
        await guarded("Mikrofon", mic_check),
        await guarded("Głośnik", speaker_check),
        await guarded("Ohbot", robot_check),
    ]
    return results
