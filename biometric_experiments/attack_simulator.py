"""
attack_simulator.py
─────────────────────────────────────────────────────────────────────────────
Attack Simulation Module — Penetration & Vulnerability Testing Framework
Chapter 5 — Security Evaluation

Provides three attack families:
  1. KeystrokeAttacker  — replay, synthetic timing, statistical morphing
  2. VoiceAttacker      — replay, pitch/time modification, synthetic voice
  3. MultimodalAttacker — coordinated attacks on both channels simultaneously

All attacks generate feature vectors in the same format as the production
pipeline so they can be fed directly into predict_keystroke() / predict_voice().

IMPORTANT: This module is for academic security evaluation only.
           All attacks are simulated in feature space — no audio or
           keystroke hardware is required or accessed.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from typing import List, Dict, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS — copied from production to keep feature space identical
# ─────────────────────────────────────────────────────────────────────────────

KEYSTROKE_HUMAN_RANGES = {
    'dwell_mean':    (40,   250),  'dwell_std':     (5,    80),
    'dwell_median':  (40,   250),  'dwell_min':     (20,   120),
    'dwell_max':     (80,   500),
    'flight_mean':   (30,   400),  'flight_std':    (10,   150),
    'flight_median': (30,   400),
    'p2p_mean':      (80,   600),  'p2p_std':       (20,   200),
    'r2r_mean':      (80,   600),  'r2r_std':       (20,   200),
    'typing_speed_cpm':  (80,  600),  'typing_duration':   (3,   30),
    'rhythm_mean':       (80,  600),  'rhythm_std':        (20,  200),
    'rhythm_cv':         (0.1, 1.5),
    'pause_count':       (0,   8),   'pause_mean':        (0,   500),
    'backspace_ratio':   (0,   0.3), 'backspace_count':   (0,   5),
    'hand_alternation_ratio':  (0.2, 0.8),
    'same_hand_sequence_mean': (1.0, 5.0),
    'finger_transition_ratio': (0.3, 0.9),
    'seek_time_mean':    (0,   300), 'seek_time_count':   (0,   5),
    'dwell_mean_norm':   (0.5, 2.0), 'dwell_std_norm':    (0.3, 1.5),
    'flight_mean_norm':  (0.5, 2.5), 'flight_std_norm':   (0.3, 2.0),
    'p2p_std_norm':      (0.3, 2.0), 'r2r_mean_norm':     (0.5, 2.5),
    'shift_lag_norm':    (0.0, 1.5),
}
DIGRAPH_RANGE = (20, 300)

VOICE_HUMAN_RANGES = {
    52: (80, 320),   # pitch_mean Hz
    53: (5,  60),    # pitch_std
    54: (1.0, 8.0),  # speaking_rate syllables/s
    55: (0.005, 0.15),  # energy_mean
    56: (0.002, 0.08),  # energy_std
    57: (0.03, 0.35),   # zcr_mean
    58: (800, 4500),    # spectral_centroid_mean
    59: (1500, 7000),   # spectral_rolloff_mean
    60: (5.0, 80.0),    # spectral_flux_mean
    61: (0.3, 0.95),    # voiced_fraction
}


# ─────────────────────────────────────────────────────────────────────────────
#  ATTACK RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

class AttackResult:
    """Container for a single attack attempt result."""

    def __init__(
        self,
        attack_type: str,
        modality: str,
        feature_vector: np.ndarray,
        metadata: dict = None,
    ):
        self.attack_type    = attack_type
        self.modality       = modality
        self.feature_vector = feature_vector
        self.metadata       = metadata or {}

    def __repr__(self):
        return (f"AttackResult(type={self.attack_type!r}, "
                f"modality={self.modality!r}, "
                f"vec_shape={self.feature_vector.shape})")


# ─────────────────────────────────────────────────────────────────────────────
#  1. KEYSTROKE ATTACKER
# ─────────────────────────────────────────────────────────────────────────────

class KeystrokeAttacker:
    """
    Simulates three classes of keystroke impostor attacks:

    replay_attack()        — exact copy of an enrollment sample (worst case:
                             attacker captured a stored feature vector)
    synthetic_variation()  — near-miss: genuine profile + small controlled noise
                             simulates an attacker who observed the target typing
    statistical_morph()    — attacker adapts their own profile toward the target
                             by blending their vector with the genuine mean
    random_impostor()      — blind attack: random human-plausible timing

    All methods return a list of AttackResult objects.
    """

    def __init__(
        self,
        profile_mean: np.ndarray,
        profile_std: np.ndarray,
        feat_names: List[str],
        rng_seed: int = 0,
    ):
        self.profile_mean = np.asarray(profile_mean, dtype=np.float64)
        self.profile_std  = np.asarray(profile_std,  dtype=np.float64)
        self.feat_names   = feat_names
        self.rng          = np.random.default_rng(rng_seed)
        self._idx         = {n: i for i, n in enumerate(feat_names)}

    # ── Attack 1: Replay ─────────────────────────────────────────────────────

    def replay_attack(
        self,
        enrollment_vectors: List[np.ndarray],
        n: int = 10,
        jitter_pct: float = 0.0,
    ) -> List[AttackResult]:
        """
        Replay attack: reuse captured enrollment feature vectors verbatim.

        Parameters
        ----------
        enrollment_vectors : genuine enrollment samples (attacker captured these)
        n                  : number of replay attempts to generate
        jitter_pct         : optional micro-jitter (0 = exact replay)
                             0.02 = ±2% noise simulating capture quantisation

        Security note: A zero-jitter replay is the hardest attack for a system
        that has no liveness detection at the feature level.
        """
        results = []
        for i in range(n):
            base = enrollment_vectors[i % len(enrollment_vectors)].copy()
            if jitter_pct > 0:
                noise = self.rng.normal(0, np.abs(base) * jitter_pct)
                base  = base + noise
            results.append(AttackResult(
                attack_type    = "keystroke_replay",
                modality       = "keystroke",
                feature_vector = base,
                metadata       = {
                    "source_idx":   i % len(enrollment_vectors),
                    "jitter_pct":   jitter_pct,
                    "description":  "Captured enrollment vector replayed verbatim",
                },
            ))
        return results

    # ── Attack 2: Synthetic Variation ─────────────────────────────────────────

    def synthetic_variation(
        self,
        n: int = 50,
        noise_pct: float = 0.05,
        target_features: Optional[List[str]] = None,
    ) -> List[AttackResult]:
        """
        Synthetic near-miss attack: draw from Gaussian centred on genuine mean
        with controlled noise.  Simulates an attacker who observed the target
        and is mimicking their timing pattern.

        Parameters
        ----------
        noise_pct        : std of noise as fraction of genuine mean (0.05 = 5%)
        target_features  : if set, only perturb these features (targeted attack)
        """
        results = []
        for _ in range(n):
            vec   = self.profile_mean.copy()
            noise = self.rng.normal(0, np.abs(self.profile_mean) * noise_pct)

            if target_features:
                # Targeted: only deviate on non-targeted features
                for fname in target_features:
                    idx = self._idx.get(fname)
                    if idx is not None:
                        noise[idx] = 0.0  # keep targeted features exact

            vec = vec + noise
            vec = np.clip(vec, 0, None)

            results.append(AttackResult(
                attack_type    = "keystroke_synthetic_variation",
                modality       = "keystroke",
                feature_vector = vec,
                metadata       = {
                    "noise_pct":      noise_pct,
                    "targeted_feats": target_features,
                    "description":    "Near-miss: genuine mean ± controlled noise",
                },
            ))
        return results

    # ── Attack 3: Statistical Morphing ────────────────────────────────────────

    def statistical_morph(
        self,
        attacker_vector: np.ndarray,
        blend_ratio: float = 0.5,
        n: int = 30,
    ) -> List[AttackResult]:
        """
        Morphing attack: attacker adapts their own typing profile toward the
        target by blending (interpolating) their vector with the genuine mean.

        blend_ratio = 0.0 → pure attacker profile
        blend_ratio = 1.0 → pure genuine mean
        blend_ratio = 0.5 → halfway (most realistic attack scenario)

        This models an attacker who has seen the target type and consciously
        tries to slow down/speed up to match.
        """
        results = []
        attacker = np.asarray(attacker_vector, dtype=np.float64)

        for i in range(n):
            # Vary blend ratio slightly per attempt to simulate trial-and-error
            alpha = np.clip(blend_ratio + self.rng.normal(0, 0.05), 0.0, 1.0)
            vec   = (1.0 - alpha) * attacker + alpha * self.profile_mean
            # Add small noise to make each attempt unique
            vec   = vec + self.rng.normal(0, self.profile_std * 0.03)
            vec   = np.clip(vec, 0, None)

            results.append(AttackResult(
                attack_type    = "keystroke_morph",
                modality       = "keystroke",
                feature_vector = vec,
                metadata       = {
                    "blend_ratio":  float(alpha),
                    "description":  f"Morphed attacker→genuine at α={alpha:.2f}",
                },
            ))
        return results

    # ── Attack 4: Random Impostor ─────────────────────────────────────────────

    def random_impostor(self, n: int = 50) -> List[AttackResult]:
        """
        Blind random attack: uniformly sample the human-plausible feature space.
        Baseline attack that a random person would produce.
        """
        results = []
        for _ in range(n):
            vec = np.zeros(len(self.feat_names))
            for i, name in enumerate(self.feat_names):
                if name in KEYSTROKE_HUMAN_RANGES:
                    lo, hi = KEYSTROKE_HUMAN_RANGES[name]
                elif name.startswith("digraph_") or name.startswith("extra_"):
                    lo, hi = DIGRAPH_RANGE
                else:
                    lo = self.profile_mean[i] * 0.3
                    hi = self.profile_mean[i] * 2.0 + 1e-6
                vec[i] = self.rng.uniform(lo, hi)

            results.append(AttackResult(
                attack_type    = "keystroke_random",
                modality       = "keystroke",
                feature_vector = vec,
                metadata       = {"description": "Blind random human-range attack"},
            ))
        return results


# ─────────────────────────────────────────────────────────────────────────────
#  2. VOICE ATTACKER
# ─────────────────────────────────────────────────────────────────────────────

class VoiceAttacker:
    """
    Simulates four classes of voice impostor attacks in MFCC feature space:

    replay_attack()        — captured enrollment MFCC vector reused
    pitch_shift_attack()   — voice shifted to match target's pitch range
    time_stretch_attack()  — speaking rate modified to match target
    synthetic_voice()      — fully synthetic MFCC drawn near genuine profile

    All attacks operate on the 62-element CMVN feature vector used by the
    production voice model (train_voice_cnn.py).
    """

    def __init__(
        self,
        raw_profile_mean: np.ndarray,  # 36-element non-CMVN profile
        raw_profile_std: np.ndarray,
        cmvn_profile_mean: np.ndarray,  # 62-element CMVN profile
        cmvn_profile_std: np.ndarray,
        rng_seed: int = 1,
    ):
        self.raw_mean  = np.asarray(raw_profile_mean,  dtype=np.float64)
        self.raw_std   = np.asarray(raw_profile_std,   dtype=np.float64)
        self.cmvn_mean = np.asarray(cmvn_profile_mean, dtype=np.float64)
        self.cmvn_std  = np.asarray(cmvn_profile_std,  dtype=np.float64)
        self.rng       = np.random.default_rng(rng_seed)

    # ── Attack 1: Voice Replay ────────────────────────────────────────────────

    def replay_attack(
        self,
        enrollment_vectors: List[np.ndarray],
        n: int = 10,
        jitter_pct: float = 0.0,
    ) -> List[AttackResult]:
        """
        Voice replay attack: reuse a captured MFCC enrollment vector.

        In a real system this would correspond to playing back a recording.
        Here we operate in feature space: the attacker obtained the stored
        MFCC template (e.g. via database breach) and replays it directly.

        jitter_pct: simulates quantisation/compression artefacts (0 = exact)
        """
        results = []
        for i in range(n):
            base = np.asarray(
                enrollment_vectors[i % len(enrollment_vectors)],
                dtype=np.float64
            ).copy()
            if jitter_pct > 0:
                base = base + self.rng.normal(0, np.abs(base) * jitter_pct)
            results.append(AttackResult(
                attack_type    = "voice_replay",
                modality       = "voice",
                feature_vector = base,
                metadata       = {
                    "source_idx":  i % len(enrollment_vectors),
                    "jitter_pct":  jitter_pct,
                    "description": "Captured MFCC template replayed verbatim",
                },
            ))
        return results

    # ── Attack 2: Pitch-Shift Attack ──────────────────────────────────────────

    def pitch_shift_attack(
        self,
        attacker_vector: np.ndarray,
        target_pitch_mean: Optional[float] = None,
        semitone_range: float = 3.0,
        n: int = 40,
    ) -> List[AttackResult]:
        """
        Pitch-shifting attack: attacker modifies their voice pitch to match
        the target's pitch range (e.g. using voice-changer software).

        In feature space: indices 52 (pitch_mean) and 53 (pitch_std) in the
        62-element CMVN vector are shifted toward the target's values.
        MFCC coefficients are also slightly perturbed (pitch affects formants).

        Parameters
        ----------
        target_pitch_mean : genuine user's mean pitch (Hz); uses profile if None
        semitone_range    : ±N semitones variation around target pitch
        """
        if target_pitch_mean is None:
            target_pitch_mean = float(self.cmvn_mean[52])

        results = []
        attacker = np.asarray(attacker_vector, dtype=np.float64).copy()

        for _ in range(n):
            vec = attacker.copy()

            # Shift pitch to target ± random semitones
            semitones  = self.rng.uniform(-semitone_range, semitone_range)
            pitch_factor = 2.0 ** (semitones / 12.0)
            vec[52] = float(np.clip(target_pitch_mean * pitch_factor, 50, 500))
            vec[53] = float(np.clip(attacker[53] * abs(pitch_factor - 0.5 + 1.0),
                                    3, 80))

            # Pitch shift slightly affects MFCCs (higher harmonics)
            mfcc_noise = self.rng.normal(0, np.abs(self.cmvn_std[:13]) * 0.15)
            vec[:13]   = vec[:13] + mfcc_noise

            results.append(AttackResult(
                attack_type    = "voice_pitch_shift",
                modality       = "voice",
                feature_vector = vec,
                metadata       = {
                    "target_pitch":   target_pitch_mean,
                    "semitone_shift": float(semitones),
                    "pitch_factor":   float(pitch_factor),
                    "description":    "Voice pitch shifted to match target",
                },
            ))
        return results

    # ── Attack 3: Time-Stretch Attack ─────────────────────────────────────────

    def time_stretch_attack(
        self,
        attacker_vector: np.ndarray,
        target_rate: Optional[float] = None,
        n: int = 40,
    ) -> List[AttackResult]:
        """
        Time-stretching attack: attacker slows down or speeds up their speech
        to match the target's speaking rate.

        In feature space: index 54 (speaking_rate) is shifted toward target's
        value. Delta-MFCCs (indices 26-38) are scaled inversely (slower speech
        = smaller deltas between frames).
        """
        if target_rate is None:
            target_rate = float(self.cmvn_mean[54])

        results = []
        attacker = np.asarray(attacker_vector, dtype=np.float64).copy()

        for _ in range(n):
            vec          = attacker.copy()
            rate_noise   = self.rng.normal(0, 0.3)
            new_rate     = float(np.clip(target_rate + rate_noise, 1.0, 10.0))
            attacker_rate = float(max(attacker[54], 0.1))
            scale        = new_rate / attacker_rate

            vec[54] = new_rate  # speaking_rate
            # Delta-MFCCs scale with tempo
            vec[26:39] = np.clip(attacker[26:39] * scale,
                                 -20.0, 20.0)

            results.append(AttackResult(
                attack_type    = "voice_time_stretch",
                modality       = "voice",
                feature_vector = vec,
                metadata       = {
                    "target_rate":   target_rate,
                    "achieved_rate": new_rate,
                    "stretch_scale": float(scale),
                    "description":   "Speaking rate time-stretched to match target",
                },
            ))
        return results

    # ── Attack 4: Synthetic Voice ─────────────────────────────────────────────

    def synthetic_voice(
        self,
        n: int = 50,
        proximity: float = 0.5,
    ) -> List[AttackResult]:
        """
        Synthetic voice attack: generate a MFCC vector that is drawn near
        the genuine profile without using any recorded sample.

        Models a text-to-speech or voice-cloning system that attempts to
        reproduce the target's vocal characteristics from publicly available
        audio (e.g. social media).

        proximity: 0.0 = random human voice, 1.0 = exact genuine mean
        """
        results = []
        for _ in range(n):
            # Interpolate between random human voice and genuine profile
            rand_vec = np.zeros(62)
            for i in range(62):
                if i in VOICE_HUMAN_RANGES:
                    lo, hi = VOICE_HUMAN_RANGES[i]
                    rand_vec[i] = self.rng.uniform(lo, hi)
                elif i < 13:
                    rand_vec[i] = self.rng.normal(self.cmvn_mean[i],
                                                  abs(self.cmvn_std[i]) * 2)
                else:
                    rand_vec[i] = self.rng.normal(0, abs(self.cmvn_std[i]) * 2)

            vec = (1.0 - proximity) * rand_vec + proximity * self.cmvn_mean
            # Add realistic noise
            vec = vec + self.rng.normal(0, self.cmvn_std * 0.10)

            results.append(AttackResult(
                attack_type    = "voice_synthetic",
                modality       = "voice",
                feature_vector = vec,
                metadata       = {
                    "proximity":   proximity,
                    "description": f"Synthetic voice at proximity={proximity:.2f} to genuine",
                },
            ))
        return results


# ─────────────────────────────────────────────────────────────────────────────
#  3. MULTIMODAL ATTACKER
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalAttacker:
    """
    Coordinates simultaneous attacks on both keystroke and voice channels.

    In a multimodal AND-gate system, an attacker must defeat BOTH factors.
    This class generates paired (keystroke, voice) attack vectors and tests
    whether coordinated attacks are more effective than independent ones.

    Attack strategies:
      coordinated_best()  — pair the best individual attack from each channel
      independent_pair()  — pair random attacks from each channel independently
      sequential_probe()  — try all combinations to find weakest pairing
    """

    def __init__(
        self,
        ks_attacker: KeystrokeAttacker,
        voice_attacker: VoiceAttacker,
    ):
        self.ks    = ks_attacker
        self.voice = voice_attacker

    def coordinated_best(
        self,
        ks_enrollment: List[np.ndarray],
        voice_enrollment: List[np.ndarray],
        n: int = 30,
    ) -> List[Tuple[AttackResult, AttackResult]]:
        """
        Best-of-both attack: use replay (hardest attack) on both channels
        simultaneously.  Represents a sophisticated attacker who obtained
        both keystroke and voice templates.
        """
        ks_attacks    = self.ks.replay_attack(ks_enrollment, n=n, jitter_pct=0.01)
        voice_attacks = self.voice.replay_attack(voice_enrollment, n=n, jitter_pct=0.01)
        return list(zip(ks_attacks, voice_attacks))

    def independent_pair(
        self,
        ks_enrollment: List[np.ndarray],
        n: int = 50,
    ) -> List[Tuple[AttackResult, AttackResult]]:
        """
        Independent-channel attack: attacker uses synthetic keystroke variation
        combined with a synthetic voice (different sophistication per channel).
        """
        ks_attacks    = self.ks.synthetic_variation(n=n, noise_pct=0.08)
        voice_attacks = self.voice.synthetic_voice(n=n, proximity=0.4)
        return list(zip(ks_attacks, voice_attacks))

    def sequential_probe(
        self,
        ks_attacks: List[AttackResult],
        voice_attacks: List[AttackResult],
    ) -> List[Tuple[AttackResult, AttackResult]]:
        """
        Exhaustive probe: try all (ks, voice) attack pairings.
        Used to find the weakest combination for threshold sensitivity analysis.
        """
        pairs = []
        for ka in ks_attacks:
            for va in voice_attacks:
                pairs.append((ka, va))
        return pairs