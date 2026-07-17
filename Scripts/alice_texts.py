from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TEMP_DIR = PROJECT_ROOT / "temp"
TEMP_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class AliceText:
    label: str
    title: str
    path: Path

    def read(self) -> str:
        return self.path.read_text(encoding="utf-8").strip()


TEXT_OPTIONS = {
    "Texte entier": AliceText(
        label="Texte entier",
        title="Texte standardisé « Le voyage d'Alice »",
        path=PROJECT_ROOT / "texte_entier.txt",
    ),
    "Texte court": AliceText(
        label="Texte court",
        title="Version courte du texte standardisé « Le voyage d'Alice »",
        path=PROJECT_ROOT / "texte_court.txt",
    ),
}
