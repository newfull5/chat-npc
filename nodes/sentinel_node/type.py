from dataclasses import dataclass, asdict
from typing import Optional, List, Iterator, Tuple, Any


@dataclass
class GameContext:
    location: Optional[str] = None
    quest: Optional[str] = None
    hp: Optional[int] = None
    mp: Optional[int] = None
    status: Optional[str] = None
    nearby: Optional[str] = None
    event_flags: Optional[List[str]] = None
    recent_action: Optional[str] = None

    def items(self):
        return asdict(self).items()

    def keys(self):
        return asdict(self).keys()

    def values(self):
        return asdict(self).values()

    @classmethod
    def get_important_keys(cls) -> List[str]:
        return [
            "location", "quest", "hp", "mp", "status",
            "nearby", "event_flags", "recent_action"
        ]

    def serialize(self) -> str:
        parts = []

        # 중요한 키들 먼저 처리
        for key in self.get_important_keys():
            value = getattr(self, key)
            if value is not None:
                parts.append(f"{key}: {value}")

        return ";\n".join(parts)


@dataclass
class EmotionResult:
    detected_emotion: str
    emotion_score: float
    emotion_changed: bool
    previous_emotion: Optional[str] = None
