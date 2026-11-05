from __future__ import annotations

import hashlib
import random

from .contracts import VOICE_LINES
from .model import VoiceVariation


ACCENT_PROFILES = {
    "neutral_west": ("neutral_a", "neutral_b"),
    "los_santos_local": ("ls_local_a", "ls_local_b"),
    "bay_area": ("bay_a", "bay_b"),
    "desert_nevada": ("desert_a", "desert_b"),
}


def stable_line_seed(world_seed: int, line_id: str) -> int:
    digest = hashlib.sha256(f"{world_seed}:{line_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def generate_voice_variations(world_seed: int, chance: float) -> list[VoiceVariation]:
    chance = max(0.0, min(1.0, chance))
    variations: list[VoiceVariation] = []
    profile_names = tuple(ACCENT_PROFILES)

    for line_id, speaker, line_text, category, source_audio in VOICE_LINES:
        random_source = random.Random(stable_line_seed(world_seed, line_id))
        varied = random_source.random() < chance

        if varied:
            accent_profile = random_source.choice(profile_names)
            voice_bank = random_source.choice(ACCENT_PROFILES[accent_profile])
            pitch = round(random_source.uniform(-2.4, 2.4), 2)
            tempo = round(random_source.uniform(0.92, 1.08), 3)
            grit = round(random_source.uniform(0.0, 0.28), 3)
        else:
            accent_profile = "original"
            voice_bank = "original"
            pitch = 0.0
            tempo = 1.0
            grit = 0.0

        variations.append(
            VoiceVariation(
                line_id=line_id,
                speaker=speaker,
                text=line_text,
                varied=varied,
                accent_profile=accent_profile,
                voice_bank=voice_bank,
                pitch_semitones=pitch,
                tempo=tempo,
                grit=grit,
                source_audio=source_audio,
            )
        )

    return variations


def make_ffmpeg_filter(variation: VoiceVariation) -> str:
    pitch_ratio = 2.0 ** (variation.pitch_semitones / 12.0)
    sample_rate = 44100.0 * pitch_ratio
    compensate_tempo = variation.tempo / pitch_ratio
    filters = [
        f"asetrate={sample_rate:.3f}",
        "aresample=44100",
        f"atempo={compensate_tempo:.6f}",
    ]
    if variation.grit > 0.0:
        drive = 1.0 + variation.grit * 4.0
        filters.append(f"volume={drive:.3f}")
        filters.append("alimiter=limit=0.92")
    return ",".join(filters)
