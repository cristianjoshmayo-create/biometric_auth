# ml/benchmark_voice_librispeech.py
#
# External voice benchmark on LibriSpeech test-clean (40 unseen speakers).
# Mirrors benchmark_voice.py but on real public data so the thesis can defend
# "ECAPA-TDNN was the most suitable algorithm" with a result that doesn't
# depend on the small 11-user enrolled cohort.
#
# Algorithms compared:
#   - MFCC mean + Cosine to user mean
#   - ECAPA-TDNN embedding + Cosine to user mean (production)
#
# Protocol per speaker:
#   * Pick first N utterances (default 10)
#   * Compute ECAPA embedding + MFCC mean per utterance
#   * LOOCV: each genuine = cosine to mean of OTHER 9 genuine
#   * Impostors = every utterance from every other speaker, scored against
#     the full genuine mean
#
# Run:  venv310/Scripts/python.exe ml/benchmark_voice_librispeech.py
#
# Output: results/voice_benchmark_librispeech.csv (per-speaker)
#         results/voice_benchmark_librispeech_summary.csv (ranked)

import os
import sys
import csv
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import librosa
import soundfile as sf

# IMPORTANT: do NOT import voice_ecapa (or speechbrain) at module load time.
# speechbrain registers lazy modules whose presence triggers a k2 import the
# next time librosa.feature.mfcc walks the call stack — silently breaking MFCC
# extraction. We import the ECAPA encoder lazily, AFTER all MFCC work is done.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sklearn.metrics import roc_auc_score, roc_curve

DATASET_DIR  = os.path.join(HERE, 'datasets', 'LibriSpeech', 'test-clean')
RESULTS_DIR  = os.path.join(os.path.dirname(HERE), 'results')
N_UTTERANCES = 10        # per speaker
MAX_SECONDS  = 8.0       # cap each clip to keep runtime manageable
SAMPLE_RATE  = 16000


def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2), float(fpr[idx]), float(fnr[idx])


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def list_speaker_utterances(speaker_dir, n):
    """Return up to n FLAC paths for this speaker, walked across chapters."""
    flacs = []
    for chapter in sorted(os.listdir(speaker_dir)):
        cdir = os.path.join(speaker_dir, chapter)
        if not os.path.isdir(cdir):
            continue
        for f in sorted(os.listdir(cdir)):
            if f.endswith('.flac'):
                flacs.append(os.path.join(cdir, f))
                if len(flacs) >= n:
                    return flacs
    return flacs


def load_audio(path):
    audio, sr = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    max_samples = int(MAX_SECONDS * SAMPLE_RATE)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    return audio


def compute_mfcc_mean(audio, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=n_mfcc)  # (13, T)
    return mfcc.mean(axis=1)  # (13,)


def loocv_cosine_to_mean(genuine, impostors):
    """genuine: (n, d); impostors: (m, d)."""
    y_t, y_s = [], []
    for i in range(len(genuine)):
        ref = np.delete(genuine, i, axis=0)
        if len(ref) == 0:
            continue
        ref_mean = ref.mean(axis=0)
        y_t.append(1); y_s.append(cosine(genuine[i], ref_mean))
    if len(genuine) >= 2 and len(impostors) > 0:
        ref_mean_full = genuine.mean(axis=0)
        for v in impostors:
            y_t.append(0); y_s.append(cosine(v, ref_mean_full))
    return np.array(y_t), np.array(y_s)


def main():
    if not os.path.isdir(DATASET_DIR):
        print(f"❌ Dataset not found at {DATASET_DIR}")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows_path    = os.path.join(RESULTS_DIR, 'voice_benchmark_librispeech.csv')
    summary_path = os.path.join(RESULTS_DIR, 'voice_benchmark_librispeech_summary.csv')

    speakers = sorted([
        s for s in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, s))
    ])
    print(f"Found {len(speakers)} speakers in {DATASET_DIR}\n")

    # ── Pass 1: MFCC extraction (must run BEFORE speechbrain is imported) ────
    print(f"\n[Pass 1/2] Extracting MFCC means ({N_UTTERANCES} utts × "
          f"{len(speakers)} speakers, ≤{MAX_SECONDS}s each)")
    spk_audio = {}     # cache decoded audio so pass 2 doesn't re-read FLAC
    spk_mfcc  = {}
    t0 = time.time()
    for si, spk in enumerate(speakers):
        utts = list_speaker_utterances(os.path.join(DATASET_DIR, spk), N_UTTERANCES)
        if len(utts) < 3:
            print(f"  [{si+1}/{len(speakers)}] {spk}: only {len(utts)} utterances, skipping")
            continue
        audio_list = []
        mfcc_vecs  = []
        for u in utts:
            try:
                audio = load_audio(u)
                mfcc_vecs.append(compute_mfcc_mean(audio).astype(np.float64))
                audio_list.append(audio)
            except Exception as e:
                print(f"     skip {os.path.basename(u)}: {e}")
        if len(mfcc_vecs) >= 3:
            spk_mfcc[spk]  = np.array(mfcc_vecs)
            spk_audio[spk] = audio_list
        if (si + 1) % 5 == 0:
            print(f"  [{si+1}/{len(speakers)}] {spk}: {len(mfcc_vecs)} clips "
                  f"({time.time()-t0:.0f}s)")
    print(f"  MFCC pass done in {time.time()-t0:.0f}s — "
          f"{len(spk_mfcc)} speakers usable")

    # ── Pass 2: ECAPA embeddings (now safe to import speechbrain) ────────────
    print(f"\n[Pass 2/2] Loading ECAPA-TDNN encoder ...")
    from voice_ecapa import _get_encoder, extract_embedding
    enc = _get_encoder()
    if enc is None:
        print("❌ ECAPA encoder unavailable — abort.")
        return

    spk_ecapa = {}
    t1 = time.time()
    for si, spk in enumerate(sorted(spk_audio.keys())):
        ecapa_vecs = []
        for audio in spk_audio[spk]:
            try:
                emb = extract_embedding(audio, sr=SAMPLE_RATE)
                if emb is None:
                    continue
                ecapa_vecs.append(np.asarray(emb, dtype=np.float64))
            except Exception as e:
                print(f"     ECAPA skip {spk}: {e}")
        if len(ecapa_vecs) >= 3:
            spk_ecapa[spk] = np.array(ecapa_vecs)
        if (si + 1) % 5 == 0:
            print(f"  [{si+1}/{len(spk_audio)}] {spk}: {len(ecapa_vecs)} embeddings "
                  f"({time.time()-t1:.0f}s)")
    print(f"  ECAPA pass done in {time.time()-t1:.0f}s — "
          f"{len(spk_ecapa)} speakers usable\n")

    # Free audio cache before scoring
    spk_audio.clear()

    # Use only speakers present in BOTH dicts so per-speaker rows align
    common = sorted(set(spk_mfcc.keys()) & set(spk_ecapa.keys()))
    spk_mfcc  = {k: spk_mfcc[k]  for k in common}
    spk_ecapa = {k: spk_ecapa[k] for k in common}
    print(f"Speakers with both MFCC and ECAPA features: {len(common)}\n")

    # ── Pass 2: per-speaker LOOCV scoring ─────────────────────────────────────
    all_rows = []

    def add(spk, algo, y_t, y_s, n_genuine, n_impostor):
        if len(y_t) == 0 or len(np.unique(y_t)) < 2:
            return
        eer, far, frr = compute_eer(y_t, y_s)
        auc = roc_auc_score(y_t, y_s)
        all_rows.append({
            'speaker': spk, 'algorithm': algo,
            'n_genuine': n_genuine, 'n_impostor': n_impostor,
            'eer': eer, 'far_at_eer': far, 'frr_at_eer': frr, 'auc': auc,
        })

    print("Per-speaker LOOCV scoring ...\n")
    for spk in spk_ecapa:
        gen_e = spk_ecapa[spk]
        imp_e = np.vstack([v for k, v in spk_ecapa.items() if k != spk])
        gen_m = spk_mfcc[spk]
        imp_m = np.vstack([v for k, v in spk_mfcc.items() if k != spk])

        yt, ys = loocv_cosine_to_mean(gen_e, imp_e)
        add(spk, 'ECAPA + Cosine', yt, ys, len(gen_e), len(imp_e))

        yt, ys = loocv_cosine_to_mean(gen_m, imp_m)
        add(spk, 'MFCC + Cosine', yt, ys, len(gen_m), len(imp_m))

        e_row = [r for r in all_rows[-2:] if r['algorithm'] == 'ECAPA + Cosine']
        m_row = [r for r in all_rows[-2:] if r['algorithm'] == 'MFCC + Cosine']
        e_eer = e_row[0]['eer'] if e_row else float('nan')
        m_eer = m_row[0]['eer'] if m_row else float('nan')
        print(f"  {spk:8s}  ECAPA EER={e_eer:.4f}   MFCC EER={m_eer:.4f}")

    # ── Save per-speaker rows ─────────────────────────────────────────────────
    if not all_rows:
        print("❌ No results.")
        return

    fields = list(all_rows[0].keys())
    with open(rows_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    # ── Aggregated summary ────────────────────────────────────────────────────
    by_algo = {}
    for r in all_rows:
        by_algo.setdefault(r['algorithm'], []).append(r)
    ranked = []
    for algo, rs in by_algo.items():
        eers = [r['eer'] for r in rs]
        aucs = [r['auc'] for r in rs]
        ranked.append((algo, len(rs), float(np.mean(eers)),
                       float(np.std(eers)), float(np.mean(aucs))))
    ranked.sort(key=lambda x: x[2])

    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['algorithm', 'n_speakers', 'mean_eer', 'std_eer', 'mean_auc'])
        for r in ranked:
            w.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}", f"{r[4]:.4f}"])

    print(f"\nPer-speaker rows -> {rows_path}")
    print(f"Summary (ranked by EER) -> {summary_path}\n")
    print(f"{'Algorithm':<22} {'Speakers':>9} {'meanEER':>9} {'stdEER':>9} {'meanAUC':>9}")
    for r in ranked:
        print(f"{r[0]:<22} {r[1]:>9d} {r[2]:>9.4f} {r[3]:>9.4f} {r[4]:>9.4f}")


if __name__ == "__main__":
    main()
