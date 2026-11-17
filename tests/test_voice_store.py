import tempfile
import unittest
from pathlib import Path

from fish_speech_api.services.voice_store import VoiceStore


class VoiceStoreTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.custom_dir = root / "voices"
        self.builtin_dir = root / "examples"
        self.store = VoiceStore(self.custom_dir, self.builtin_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def wav_bytes():
        return b"RIFF\x04\x00\x00\x00WAVE"

    def test_create_and_load_voice(self):
        name = self.store.create("My-Voice", self.wav_bytes(), "  Hello there.  ")

        self.assertEqual("my-voice", name)
        self.assertEqual((self.wav_bytes(), "Hello there."), self.store.get("my-voice"))
        self.assertEqual(0o600, self.custom_dir.joinpath("my-voice.wav").stat().st_mode & 0o777)

    def test_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            self.store.create("../voice", self.wav_bytes(), "Hello")

    def test_rejects_invalid_wav(self):
        with self.assertRaisesRegex(ValueError, "must be WAV"):
            self.store.create("voice", b"not a wav file", "Hello")

    def test_does_not_overwrite_existing_voice(self):
        self.store.create("voice", self.wav_bytes(), "First")

        with self.assertRaises(FileExistsError):
            self.store.create("voice", self.wav_bytes(), "Second")

        self.assertEqual("First", self.store.get("voice")[1])

    def test_reads_builtin_voice(self):
        self.builtin_dir.mkdir()
        self.builtin_dir.joinpath("english.wav").write_bytes(self.wav_bytes())
        self.builtin_dir.joinpath("english.lab").write_text("Hello", encoding="utf-8")

        self.assertEqual((self.wav_bytes(), "Hello"), self.store.get("english"))

    def test_missing_voice_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.store.get("missing")


if __name__ == "__main__":
    unittest.main()
