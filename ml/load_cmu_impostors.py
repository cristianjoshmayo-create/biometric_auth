# ml/load_cmu_impostors.py
#
# Downloads the CMU Keystroke Dynamics Benchmark dataset and converts
# each of the 51 users into a feature vector compatible with your
# train_keystroke_rf.py FEATURE_NAMES.
#
# Usage:
#   python ml/load_cmu_impostors.py
#
# Output:
#   ml/models/cmu_impostor_profiles.pkl
#   — a list of 51 numpy arrays, one per CMU subject,
#     ready to be passed as real_impostors in train_keystroke_rf.py
#
# The CMU dataset uses password ".tie5Roanl" so the digraph columns
# won't match your passphrase. We set all digraph features to 0 for
# CMU users and rely on the timing + behavioral features instead.
# That is still highly discriminative — dwell/flight/rhythm are the
# most important features anyway.

import os
import sys
import pickle
import numpy as np

# ── Download ─────────────────────────────────────────────────────────────────
CMU_URL  = "https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CSV_PATH = os.path.join(DATA_DIR, "DSL-StrongPasswordData.csv")

def download_cmu():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(CSV_PATH):
        print(f"  Already downloaded: {CSV_PATH}")
        return
    print(f"  Downloading CMU dataset from {CMU_URL} ...")
    import urllib.request
    urllib.request.urlretrieve(CMU_URL, CSV_PATH)
    print(f"  Saved to {CSV_PATH}")


# ── CMU column layout ─────────────────────────────────────────────────────────
# The CSV has columns:
#   subject, sessionIndex, rep,
#   H.period, DD.period.t, UD.period.t,
#   H.t, DD.t.i, UD.t.i,
#   H.i, DD.i.e, UD.i.e,
#   H.e, DD.e.five, UD.e.five,
#   H.five, DD.five.Shift.r, UD.five.Shift.r,
#   H.Shift.r, DD.Shift.r.o, UD.Shift.r.o,
#   H.o, DD.o.a, UD.o.a,
#   H.a, DD.a.n, UD.a.n,
#   H.n, DD.n.l, UD.n.l,
#   H.l, DD.l.Return, UD.l.Return,
#   H.Return
#
# H.*   = Hold time (dwell) for each key
# DD.*  = Down-Down time (press-to-press = p2p) between consecutive keys
# UD.*  = Up-Down time (flight time) between consecutive keys

def load_cmu_csv():
    import csv
    rows = []
    with open(CSV_PATH, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"  Loaded {len(rows)} rows from CMU CSV")
    return rows


def extract_cmu_features(rows):
    """
    Group rows by subject, compute per-subject statistical features
    that map to your FEATURE_NAMES as closely as possible.
    Returns dict: subject_id -> feature_vector (numpy array)
    """
    from collections import defaultdict
    subjects = defaultdict(list)
    for row in rows:
        subjects[row['subject']].append(row)

    # H.* columns = hold/dwell times
    H_COLS = [
        'H.period', 'H.t', 'H.i', 'H.e', 'H.five',
        'H.Shift.r', 'H.o', 'H.a', 'H.n', 'H.l', 'H.Return'
    ]
    # UD.* columns = flight times (up-down)
    UD_COLS = [
        'UD.period.t', 'UD.t.i', 'UD.i.e', 'UD.e.five',
        'UD.five.Shift.r', 'UD.Shift.r.o', 'UD.o.a', 'UD.a.n', 'UD.n.l', 'UD.l.Return'
    ]
    # DD.* columns = down-down / press-to-press
    DD_COLS = [
        'DD.period.t', 'DD.t.i', 'DD.i.e', 'DD.e.five',
        'DD.five.Shift.r', 'DD.Shift.r.o', 'DD.o.a', 'DD.a.n', 'DD.n.l', 'DD.l.Return'
    ]

    profiles = {}
    for subj, subj_rows in subjects.items():
        # Collect all samples for this subject
        all_hold   = []
        all_flight = []
        all_p2p    = []

        for row in subj_rows:
            try:
                hold   = [float(row[c]) * 1000 for c in H_COLS]   # convert s → ms
                flight = [float(row[c]) * 1000 for c in UD_COLS]
                p2p    = [float(row[c]) * 1000 for c in DD_COLS]
                all_hold.append(hold)
                all_flight.append(flight)
                all_p2p.append(p2p)
            except (ValueError, KeyError):
                continue

        if not all_hold:
            continue

        hold_arr   = np.array(all_hold)    # shape: (n_samples, 11)
        flight_arr = np.array(all_flight)  # shape: (n_samples, 10)
        p2p_arr    = np.array(all_p2p)     # shape: (n_samples, 10)

        # Per-sample means, then average across samples
        hold_per_sample   = hold_arr.mean(axis=1)    # mean dwell per typing attempt
        flight_per_sample = flight_arr.mean(axis=1)
        p2p_per_sample    = p2p_arr.mean(axis=1)

        dwell_mean   = float(hold_per_sample.mean())
        dwell_std    = float(hold_per_sample.std())
        dwell_median = float(np.median(hold_per_sample))
        dwell_min    = float(hold_per_sample.min())
        dwell_max    = float(hold_per_sample.max())

        flight_mean   = float(flight_per_sample.mean())
        flight_std    = float(flight_per_sample.std())
        flight_median = float(np.median(flight_per_sample))

        p2p_mean = float(p2p_per_sample.mean())
        p2p_std  = float(p2p_per_sample.std())

        # r2r ≈ p2p for fixed-text (no between-word gaps)
        r2r_mean = p2p_mean
        r2r_std  = p2p_std

        # Typing speed: password is 10 chars, time = last DD + last hold
        # Approximate from p2p mean * 9 intervals + last hold
        typing_duration  = float((p2p_mean * 9 + dwell_mean) / 1000)  # seconds
        typing_speed_cpm = float(600.0 / typing_duration) if typing_duration > 0 else 200.0

        # Rhythm: std of inter-key intervals (p2p)
        rhythm_mean = p2p_mean
        rhythm_std  = p2p_std
        rhythm_cv   = float(p2p_std / p2p_mean) if p2p_mean > 0 else 0.5

        # Normalised features (ratio to typing speed)
        speed_norm = typing_speed_cpm / 300.0 if typing_speed_cpm > 0 else 1.0
        dwell_mean_norm  = dwell_mean  / (60000 / typing_speed_cpm) if typing_speed_cpm > 0 else 1.0
        dwell_std_norm   = dwell_std   / max(dwell_mean, 1)
        flight_mean_norm = flight_mean / (60000 / typing_speed_cpm) if typing_speed_cpm > 0 else 1.0
        flight_std_norm  = flight_std  / max(flight_mean, 1)
        p2p_std_norm     = p2p_std     / max(p2p_mean, 1)
        r2r_mean_norm    = r2r_mean    / (60000 / typing_speed_cpm) if typing_speed_cpm > 0 else 1.0

        profiles[subj] = {
            'dwell_mean': dwell_mean,   'dwell_std': dwell_std,
            'dwell_median': dwell_median, 'dwell_min': dwell_min, 'dwell_max': dwell_max,
            'flight_mean': flight_mean, 'flight_std': flight_std, 'flight_median': flight_median,
            'p2p_mean': p2p_mean,       'p2p_std': p2p_std,
            'r2r_mean': r2r_mean,       'r2r_std': r2r_std,
            'typing_speed_cpm': typing_speed_cpm, 'typing_duration': typing_duration,
            'rhythm_mean': rhythm_mean, 'rhythm_std': rhythm_std, 'rhythm_cv': rhythm_cv,
            'dwell_mean_norm': dwell_mean_norm,   'dwell_std_norm': dwell_std_norm,
            'flight_mean_norm': flight_mean_norm, 'flight_std_norm': flight_std_norm,
            'p2p_std_norm': p2p_std_norm, 'r2r_mean_norm': r2r_mean_norm,
            # These can't be derived from CMU (fixed short password, no shifts in ".tie5Roanl")
            'pause_count': 0, 'pause_mean': 0,
            'backspace_ratio': 0, 'backspace_count': 0,
            'hand_alternation_ratio': 0.5,  # neutral estimate
            'same_hand_sequence_mean': 2.0,
            'finger_transition_ratio': 0.6,
            'seek_time_mean': 0, 'seek_time_count': 0,
            'shift_lag_mean': dwell_mean * 0.3,  # Shift.r key is present in CMU
            'shift_lag_std': dwell_std * 0.3,
            'shift_lag_count': 1,
            'shift_lag_norm': 0.3,
            # Digraphs: set to 0 — passphrase doesn't match
            **{f'digraph_{d}': 0.0 for d in [
                'th','he','bi','io','om','me','et','tr','ri','ic',
                'vo','oi','ce','ke','ey','ys','st','ro','ok','au',
                'ut','en','nt','ti','ca','at','on'
            ]},
        }

    return profiles


def profiles_to_vectors(profiles: dict) -> list:
    """Convert profile dicts to numpy vectors in FEATURE_NAMES order."""
    # Import FEATURE_NAMES from training script
    ml_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, ml_dir)
    from train_keystroke_rf import FEATURE_NAMES

    vectors = []
    for subj, feat in profiles.items():
        vec = np.array([
            float(feat.get(name, 0.0) or 0.0)
            for name in FEATURE_NAMES
        ], dtype=np.float64)
        vectors.append(vec)
    return vectors


def build_and_save():
    print(f"\n{'='*60}")
    print(f"  CMU IMPOSTOR PROFILE BUILDER")
    print(f"{'='*60}")

    download_cmu()
    rows     = load_cmu_csv()
    profiles = extract_cmu_features(rows)

    print(f"  Extracted profiles for {len(profiles)} subjects")

    # Print a few stats to verify sanity
    vecs = profiles_to_vectors(profiles)
    arr  = np.array(vecs)
    print(f"\n  Sanity check (mean across all 51 CMU subjects):")
    from train_keystroke_rf import FEATURE_NAMES
    for feat in ['dwell_mean', 'flight_mean', 'typing_speed_cpm', 'rhythm_cv']:
        idx = FEATURE_NAMES.index(feat)
        print(f"    {feat:20s}: {arr[:, idx].mean():.2f} ± {arr[:, idx].std():.2f}")

    # Save
    model_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(model_dir, exist_ok=True)
    output_path = os.path.join(model_dir, 'cmu_impostor_profiles.pkl')

    with open(output_path, 'wb') as f:
        pickle.dump(vecs, f)

    print(f"\n  ✅ Saved {len(vecs)} CMU impostor vectors → {output_path}")
    print(f"     These will be auto-loaded by train_keystroke_rf.py\n")
    return output_path


if __name__ == "__main__":
    # Add ml/ dir to path so train_keystroke_rf imports work
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    build_and_save()
    