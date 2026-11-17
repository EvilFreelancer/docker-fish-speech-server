import os
import re
from pathlib import Path
from typing import Tuple

MAX_AUDIO_BYTES = 25 * 1024 * 1024
VOICE_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class VoiceStore:
    def __init__(self, custom_dir: Path, builtin_dir: Path):
        self.custom_dir = Path(custom_dir)
        self.builtin_dir = Path(builtin_dir)

    @staticmethod
    def normalize_name(name: str) -> str:
        if not isinstance(name, str):
            raise ValueError("Voice name must be a string")

        normalized = name.strip().lower()
        if not VOICE_NAME_PATTERN.fullmatch(normalized):
            raise ValueError("Voice name must use 1-64 letters, numbers, underscores or hyphens")

        return normalized

    def get(self, name: str) -> Tuple[bytes, str]:
        voice_name = self.normalize_name(name)

        for directory in (self.custom_dir, self.builtin_dir):
            audio_path = directory / f"{voice_name}.wav"
            text_path = directory / f"{voice_name}.lab"
            if audio_path.exists() or text_path.exists():
                if not audio_path.is_file() or not text_path.is_file():
                    raise FileNotFoundError(f"Voice sample is incomplete: {voice_name}")

                return audio_path.read_bytes(), text_path.read_text(encoding="utf-8").strip()

        raise FileNotFoundError(f"Voice sample not found: {voice_name}")

    def create(self, name: str, audio: bytes, reference_text: str) -> str:
        voice_name = self.normalize_name(name)
        if not isinstance(audio, bytes) or len(audio) > MAX_AUDIO_BYTES:
            raise ValueError("Audio data too large (max 25MB)")
        if len(audio) < 12 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
            raise ValueError("Invalid audio format (must be WAV)")
        if not isinstance(reference_text, str) or not reference_text.strip():
            raise ValueError("Reference text is required")
        if len(reference_text) > 4096:
            raise ValueError("Reference text too long (max 4096 chars)")

        paths = [directory / f"{voice_name}.{extension}" for directory in (self.custom_dir, self.builtin_dir) for extension in ("wav", "lab")]
        if any(path.exists() for path in paths):
            raise FileExistsError(f"Voice already exists: {voice_name}")

        self.custom_dir.mkdir(parents=True, exist_ok=True)
        audio_path = self.custom_dir / f"{voice_name}.wav"
        text_path = self.custom_dir / f"{voice_name}.lab"
        created_paths = []

        try:
            self._write_exclusive(audio_path, audio)
            created_paths.append(audio_path)
            self._write_exclusive(text_path, reference_text.strip().encode("utf-8") + b"\n")
            created_paths.append(text_path)
        except Exception:
            for path in created_paths:
                path.unlink(missing_ok=True)
            raise

        return voice_name

    @staticmethod
    def _write_exclusive(path: Path, content: bytes) -> None:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
