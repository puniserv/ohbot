import json
import os
import re
import stat
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path



def app_root() -> Path:
    # the build uses --contents-directory . so web/, personas/, config.toml and .env all sit next to the exe
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


ROOT = app_root()


@dataclass
class OpenAIConfig:
    stt_model: str = "gpt-transcribe"
    chat_model: str = "gpt-5.4-mini"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "marin"
    moderation_model: str = "omni-moderation-latest"
    max_tokens: int = 200
    timeout_s: float = 10
    reasoning_effort: str = "none"  # "" omits the parameter (non-reasoning chat models reject it)


DEFAULT_FILLERS = ["Hmm…", "Chwileczkę…", "Niech pomyślę…", "Już mówię…"]


@dataclass
class ConversationConfig:
    persona: str = "dzieci"
    history_turns: int = 10
    mode: str = "ptt"
    max_record_s: float = 30  # a stuck button or dropped page must not record forever
    greeting: str = "Cześć! Jestem Ohbot. O czym porozmawiamy?"
    fillers: bool = True
    filler_phrases: list[str] = field(default_factory=lambda: list(DEFAULT_FILLERS))


@dataclass
class WakeConfig:
    words: list[str] = field(default_factory=lambda: ["ohbot", "hej ohbot", "ochbot", "obot", "ołbot"])
    window_s: float = 8  # after a bare "Ohbot!" the next utterance needs no wake word


@dataclass
class VadConfig:
    aggressiveness: int = 2
    silence_ms: int = 800
    min_speech_ms: int = 300


@dataclass
class BargeInConfig:
    enabled: bool = False
    threshold: float = 0.15
    hold_ms: int = 300


@dataclass
class ColoursConfig:
    idle: list[int] = field(default_factory=lambda: [0, 0, 3])
    listening: list[int] = field(default_factory=lambda: [0, 10, 0])
    thinking: list[int] = field(default_factory=lambda: [0, 0, 10])
    speaking: list[int] = field(default_factory=lambda: [10, 10, 10])
    error: list[int] = field(default_factory=lambda: [10, 0, 0])


@dataclass
class RobotConfig:
    # real motor direction of the lips is unverified; flip here if the mouth moves the wrong way
    lip_invert_top: bool = False
    lip_invert_bottom: bool = False
    lip_delay_ms: int = 0  # lips can lead the sound by the output latency; delay them to match
    # measured on the robot: 1 already presses the lips together, the ohbot lib's 5 leaves the mouth open
    lip_closed: float = 1.5
    lip_open: float = 6.5
    gesture_step_s: float = 0.25
    gesture_speed: int = 8
    gesture_scale: float = 1.0  # amplitude around the rest pose
    # head and eyes glide to their target each tick: rate = share of the gap, max_step = cap per 0.1 s tick
    head_rate: float = 0.3
    head_max_step: float = 0.4
    head_speed: int = 3  # servo speed of those small steps; too fast and the head stutters at 10 Hz
    eye_rate: float = 0.6


@dataclass
class CameraConfig:
    enabled: bool = True
    index: int = 0
    mirror: bool = False  # flip when the robot looks the wrong way
    offset_x: float = 0.0  # the camera sits beside the robot, not on its axis
    offset_y: float = 0.0
    head_gain: float = 3.0
    eye_gain: float = 10.0
    deadband: float = 0.5
    smoothing: float = 0.5
    min_score: float = 0.7
    lost_after_s: float = 3.0
    auto_greet: bool = True
    greet_cooldown_s: float = 60.0


@dataclass
class Config:
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    barge_in: BargeInConfig = field(default_factory=BargeInConfig)
    colours: ColoursConfig = field(default_factory=ColoursConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    api_key: str | None = None
    root: Path = ROOT


def _read_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip("\"'")
    return out


def load_config(root: Path = ROOT, env: dict[str, str] | None = None) -> Config:
    """Environment wins over `.env`; a missing key yields `api_key=None` so `doctor` can explain it."""
    env = os.environ if env is None else env
    cfg = Config(root=root)
    path = root / "config.toml"
    raw = tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    for f in fields(cfg):
        section = getattr(cfg, f.name)
        if f.name in raw and hasattr(section, "__dataclass_fields__"):
            for k, v in raw[f.name].items():
                if hasattr(section, k):
                    setattr(section, k, v)
    key = env.get("OPENAI_API_KEY") or _read_dotenv(root / ".env").get("OPENAI_API_KEY")
    cfg.api_key = key or None
    return cfg


VOICES = ("alloy", "ash", "ballad", "cedar", "coral", "echo", "fable", "marin", "nova", "onyx", "sage", "shimmer", "verse")


def _number(kind: type, lo: float, hi: float):
    def check(v):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (kind is int and v != int(v)) or not lo <= v <= hi:
            raise ValueError(f"expected {kind.__name__} in {lo}..{hi}")
        return kind(v)

    return check


def _flag(v):
    if not isinstance(v, bool):
        raise ValueError("expected true/false")
    return v


_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def _text(max_len: int):
    def check(v):
        if not isinstance(v, str) or not v.strip() or len(v) > max_len:
            raise ValueError(f"expected 1..{max_len} characters")
        if _CONTROL.search(v):  # tomllib rejects a raw DEL or control character on the next load
            raise ValueError("control characters are not allowed")
        return v.strip()

    return check


def _texts(max_items: int, max_len: int):
    one = _text(max_len)

    def check(v):
        if not isinstance(v, list) or not 1 <= len(v) <= max_items:
            raise ValueError(f"expected a list of 1..{max_items} texts")
        return [one(x) for x in v]

    return check


def _voice(v):
    if v not in VOICES:
        raise ValueError("unknown voice")
    return v


# the only keys the UI may change; each maps to a checker returning the normalised value
EDITABLE = {
    "robot": {
        "lip_closed": _number(float, 0, 10),
        "lip_open": _number(float, 0, 10),
        "lip_delay_ms": _number(int, 0, 1000),
        "lip_invert_top": _flag,
        "lip_invert_bottom": _flag,
        "gesture_step_s": _number(float, 0.05, 2),
        "gesture_speed": _number(int, 1, 10),
        "gesture_scale": _number(float, 0.5, 1.5),
        "head_rate": _number(float, 0.05, 1),
        "head_max_step": _number(float, 0.05, 3),
        "head_speed": _number(int, 1, 10),
        "eye_rate": _number(float, 0.05, 1),
    },
    "openai": {"tts_voice": _voice},
    "barge_in": {"threshold": _number(float, 0, 1)},
    "vad": {"silence_ms": _number(int, 200, 5000)},
    "wake": {"window_s": _number(float, 1, 60)},
    "conversation": {"greeting": _text(300), "fillers": _flag, "filler_phrases": _texts(10, 40)},
    "camera": {
        "enabled": _flag,
        "index": _number(int, 0, 9),
        "mirror": _flag,
        "offset_x": _number(float, -1, 1),
        "offset_y": _number(float, -1, 1),
        "head_gain": _number(float, 0, 15),
        "eye_gain": _number(float, 0, 15),
        "deadband": _number(float, 0, 3),
        "smoothing": _number(float, 0, 0.95),
        "min_score": _number(float, 0.3, 0.99),
        "lost_after_s": _number(float, 0.5, 30),
        "auto_greet": _flag,
        "greet_cooldown_s": _number(float, 10, 3600),
    },
}


def validate_settings(raw) -> dict[str, dict]:
    """Raises ValueError on anything but known editable keys with valid values."""
    if not isinstance(raw, dict):
        raise ValueError("expected an object")
    out: dict[str, dict] = {}
    for section, values in raw.items():
        checks = EDITABLE.get(section)
        if checks is None or not isinstance(values, dict):
            raise ValueError(f"unknown section {section!r}")
        for key, value in values.items():
            if key not in checks:
                raise ValueError(f"unknown key {section}.{key}")
            try:
                out.setdefault(section, {})[key] = checks[key](value)
            except ValueError as e:
                raise ValueError(f"{section}.{key}: {e}") from None
    return out


def editable_settings(cfg: Config) -> dict[str, dict]:
    return {s: {k: getattr(getattr(cfg, s), k) for k in keys} for s, keys in EDITABLE.items()}


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (str, list)):
        return json.dumps(v, ensure_ascii=False)  # a JSON string, or array of them, is valid TOML
    return repr(v)


def _parses(value: str) -> bool:
    try:
        tomllib.loads("v = " + value)
        return True
    except tomllib.TOMLDecodeError:
        return False


def _comment(rest: str) -> str:
    """The trailing comment after a value (a `#` inside a string or an unfinished array is not one)."""
    for i, c in enumerate(rest):
        if c == "#" and _parses(rest[:i]):
            return "  " + rest[i:]
    return ""


_SECTION = re.compile(r"^\s*\[([^\[\]]+)\]\s*(#.*)?$")


def _set_line(lines: list[str], section: str, key: str, line: str) -> None:
    current = header = end = None
    for i, text in enumerate(lines):
        if m := _SECTION.match(text):
            current = m.group(1).strip()
            if current == section:
                header = end = i
        elif current == section:
            if m := re.match(rf"\s*{re.escape(key)}\s*=(.*)$", text):
                value, last = m.group(1), i
                # a hand-written array over several lines; anything else broken stays a one-line replace
                while value.lstrip().startswith("[") and not _parses(value) and last + 1 < len(lines):
                    last += 1
                    value += "\n" + lines[last]
                lines[i : last + 1] = [line + _comment(value.strip())]
                return
            if text.strip():
                end = i
    if header is None:
        lines += ["", f"[{section}]", line]
    else:
        lines.insert(end + 1, line)


def save_settings(path: Path, values: dict[str, dict]) -> None:
    """Rewrites only the lines of the given keys so the comments in config.toml survive; adds missing ones."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    for section, keys in values.items():
        for key, value in keys.items():
            _set_line(lines, section, key, f"{key} = {_toml_value(value)}")
    write_atomic(path, "\n".join(lines) + "\n")


def write_atomic(path: Path, text: str) -> None:
    """A crash or full disk mid-write must leave the old file, not half of the new one."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())  # else a power cut can leave an empty config.toml that loads as all defaults
        os.chmod(tmp, stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644)  # mkstemp makes it owner-only
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise
