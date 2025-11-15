from __future__ import annotations
import os
from typing import Any, Optional, Tuple
from utils import get_env
from fish_audio_sdk import Session as FishAudioSession, TTSRequest as FishAudioTTSRequest


class SpeechService:
    def __init__(self):
        self._session: Any = None
        self._reference_id: Optional[str] = None
        self._backend: str = os.getenv("FISH_AUDIO_BACKEND", "s1")
        self._audio_format: str = os.getenv("FISH_AUDIO_FORMAT", "mp3")

    def _ensure(self) -> None:
        if self._session is None:
            api_key = get_env(["FISH_AUDIO_API_KEY"], required=True)
            self._session = FishAudioSession(api_key)

        if self._reference_id is None:
            reference_id = get_env(["FISH_AUDIO_MODEL_ID"], required=True)
            self._reference_id = reference_id

    def synthesize(
        self,
        text: str,
        *,
        backend: Optional[str] = None,
        audio_format: Optional[str] = None,
    ) -> Tuple[Optional[bytes], str]:
        fmt = (audio_format or "mp3").lower()
        self._ensure()
        if not text or not text.strip():
            return None, fmt

        if not self._reference_id:
            raise RuntimeError("Fish Audio reference id is not configured.")

        backend_name = backend or "s1"
        request = FishAudioTTSRequest(text=text.strip(), reference_id=self._reference_id, format=fmt)

        audio_buffer = bytearray()
        for chunk in self._session.tts(request, backend=backend_name):
            if chunk:
                audio_buffer.extend(chunk)

        return (bytes(audio_buffer) if audio_buffer else None, fmt)
