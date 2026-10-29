"""Audio alert and cooldown management using Pygame mixer.

Provides non-blocking, thread-safe sound playback, tone generation fallback,
and mute controls.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
import logging
import math
import struct
import threading
import wave

from config import ALARM_SOUND_PATH, SOUNDS_DIR

logger = logging.getLogger(__name__)


def generate_fallback_alarm_wav(
    target_path: Path,
    sample_rate: int = 44100,
    frequency: float = 950.0,
    duration: float = 0.8,
) -> Path:
    """Synthesize a dual-pulse square/sine alert sound and save as WAV.

    Args:
        target_path: Path to write the .wav file.
        sample_rate: Audio sampling rate in Hz.
        frequency: Base tone frequency in Hz.
        duration: Total audio duration in seconds.

    Returns:
        Path to the written WAV file.
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = int(sample_rate * duration)
    data = bytearray()

    for i in range(n_samples):
        t = i / sample_rate
        # Create pulsed beep: 150ms sound, 100ms silence
        cycle = t % 0.25
        if cycle < 0.15:
            # Dual frequency harmonic for urgency: base + 1.5x
            s1 = math.sin(2.0 * math.pi * frequency * t)
            s2 = math.sin(2.0 * math.pi * (frequency * 1.5) * t)
            amplitude = 32767.0 * 0.5 * (0.7 * s1 + 0.3 * s2)
            sample = int(amplitude)
        else:
            sample = 0
        data.extend(struct.pack("<h", sample))

    with wave.open(str(target_path), "wb") as wav_file:
        wav_file.setnchannels(1)       # Mono
        wav_file.setsampwidth(2)       # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(data)

    logger.info("Generated synthetic alarm tone at %s", target_path)
    return target_path


class AlarmManager:
    """Thread-safe, non-blocking audio alarm player with mute controls."""

    def __init__(
        self,
        sound_path: Path = ALARM_SOUND_PATH,
        auto_generate_sound: bool = True,
        initial_muted: bool = False,
        init_audio: bool = True,
    ) -> None:
        """Initialize the alarm manager.

        Args:
            sound_path: Path to the alert sound WAV file.
            auto_generate_sound: Generate a synthetic tone if file is missing.
            initial_muted: Whether alarm starts muted.
            init_audio: Whether to initialize pygame mixer (can disable for tests).
        """
        self.sound_path = Path(sound_path)
        self.is_muted = initial_muted
        self.is_audio_available = False
        self._sound = None
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="alarm_worker")

        if init_audio:
            self._init_pygame_mixer(auto_generate_sound)

    def _init_pygame_mixer(self, auto_generate: bool) -> None:
        """Initialize Pygame mixer and load sound file safely."""
        try:
            import pygame
            # Check if mixer is already initialized, else initialize
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

            # Ensure sound file exists
            if not self.sound_path.exists():
                if auto_generate:
                    generate_fallback_alarm_wav(self.sound_path)
                else:
                    logger.warning("Alarm sound file not found at %s", self.sound_path)
                    return

            self._sound = pygame.mixer.Sound(str(self.sound_path))
            self.is_audio_available = True
            logger.info("Pygame audio alarm initialized successfully with %s", self.sound_path)
        except Exception as exc:
            self.is_audio_available = False
            self._sound = None
            logger.warning(
                "Audio playback unavailable (Pygame mixer error: %s). "
                "Application will run with visual-only alerts.",
                exc,
            )

    def trigger_alarm(self) -> bool:
        """Trigger the alarm sound asynchronously in a worker thread.

        Returns:
            True if playback was dispatched, False if muted or unavailable.
        """
        if self.is_muted:
            return False

        if not self.is_audio_available or self._sound is None:
            # Visual or terminal alert fallback
            print("\a", end="", flush=True)  # ASCII bell fallback
            return False

        def _play():
            with self._lock:
                try:
                    if self._sound is not None and not self.is_muted:
                        self._sound.play()
                except Exception as exc:
                    logger.warning("Error playing alarm sound: %s", exc)

        self._executor.submit(_play)
        return True

    def toggle_mute(self) -> bool:
        """Toggle audio mute state.

        Returns:
            New mute state (True if muted, False if unmuted).
        """
        self.is_muted = not self.is_muted
        if self.is_muted:
            self.stop()
        return self.is_muted

    def stop(self) -> None:
        """Stop any currently playing sound."""
        with self._lock:
            if self._sound is not None:
                try:
                    self._sound.stop()
                except Exception:
                    pass

    def close(self) -> None:
        """Stop sounds and shutdown thread pool."""
        self.stop()
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

    def __enter__(self) -> "AlarmManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
