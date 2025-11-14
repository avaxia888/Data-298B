from __future__ import annotations

import os
from typing import Any, Optional, Tuple

from utils import get_env

try:
    from fish_audio_sdk import Session as FishAudioSession, TTSRequest as FishAudioTTSRequest
except ImportError:  # pragma: no cover - optional dependency
    FishAudioSession = None
    FishAudioTTSRequest = None


class SpeechService:
    def __init__(self):
        self._session: Any = None
        self._reference_id: Optional[str] = None
        self._backend: str = os.getenv("FISH_AUDIO_BACKEND", "s1")
        self._audio_format: str = os.getenv("FISH_AUDIO_FORMAT", "mp3")

    def _ensure(self) -> None:
        if FishAudioSession is None or FishAudioTTSRequest is None:
            raise RuntimeError("fish-audio-sdk is not installed. Install it with 'pip install fish-audio-sdk'.")

        if self._session is None:
            api_key = get_env(["FISH_AUDIO_API_KEY"], required=True)
            self._session = FishAudioSession(api_key)

        if self._reference_id is None:
            reference_id = get_env(["FISH_AUDIO_REFERENCE_ID", "FISH_AUDIO_MODEL_ID"], required=True)
            self._reference_id = reference_id

    def synthesize(
        self,
        text: str,
        *,
        reference_id: Optional[str] = None,
        backend: Optional[str] = None,
        audio_format: Optional[str] = None,
    ) -> Tuple[Optional[bytes], str]:
        fmt = (audio_format or os.getenv("FISH_AUDIO_FORMAT") or self._audio_format or "mp3").lower()
        if not text or not text.strip():
            return None, fmt

        self._ensure()

        resolved_reference = reference_id or self._reference_id
        if not resolved_reference:
            raise RuntimeError("Fish Audio reference id is not configured.")

        backend_name = backend or os.getenv("FISH_AUDIO_BACKEND") or self._backend or "s1"

        request = FishAudioTTSRequest(text=text.strip(), reference_id=resolved_reference, format=fmt)

        audio_buffer = bytearray()
        try:
            for chunk in self._session.tts(request, backend=backend_name):
                if chunk:
                    audio_buffer.extend(chunk)
        except Exception as exc:  # pragma: no cover - network/service errors
            raise RuntimeError(f"Fish Audio synthesis failed: {exc}") from exc

        return (bytes(audio_buffer) if audio_buffer else None, fmt)
