import os
import sys
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


@dataclass
class ConversationConfig:
    persona: str = "dzieci"
    history_turns: int = 10
    mode: str = "ptt"
    max_record_s: float = 30  # a stuck button or dropped page must not record forever


@dataclass
class WakeConfig:
    words: list[str] = field(default_factory=lambda: ["ohbot", "hej ohbot", "ochbot", "obot", "ołbot"])


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


@dataclass
class Config:
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    barge_in: BargeInConfig = field(default_factory=BargeInConfig)
    colours: ColoursConfig = field(default_factory=ColoursConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
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
