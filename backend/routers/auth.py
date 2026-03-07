# backend/routers/auth.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import hashlib

from database.db import get_db
from database.models import User, KeystrokeTemplate, VoiceTemplate, SecurityQuestion, AuthLog

router = APIRouter()

# ── Schemas ───────────────────────────────────────────────

class KeystrokeAuth(BaseModel):
    username: str
    dwell_times: List[float]
    flight_times: List[float]
    typing_speed: float
    
    # All 34 features (same as enrollment)
    dwell_mean: float = 0
    dwell_std: float = 0
    dwell_median: float = 0
    dwell_min: float = 0
    dwell_max: float = 0
    flight_mean: float = 0
    flight_std: float = 0
    flight_median: float = 0
    p2p_mean: float = 0
    p2p_std: float = 0
    r2r_mean: float = 0
    r2r_std: float = 0
    digraph_th: float = 0
    digraph_he: float = 0
    digraph_in: float = 0
    digraph_er: float = 0
    digraph_an: float = 0
    digraph_ed: float = 0
    digraph_to: float = 0
    digraph_it: float = 0
    typing_speed_cpm: float = 0
    typing_duration: float = 0
    rhythm_mean: float = 0
    rhythm_std: float = 0
    rhythm_cv: float = 0
    pause_count: float = 0
    pause_mean: float = 0
    backspace_ratio: float = 0
    backspace_count: float = 0
    hand_alternation_ratio: float = 0
    same_hand_sequence_mean: float = 0
    finger_transition_ratio: float = 0
    seek_time_mean: float = 0
    seek_time_count: float = 0

class VoiceAuth(BaseModel):
    username: str
    mfcc_features: List[float]

class SecurityAuth(BaseModel):
    username: str
    answer: str

# ── Auth Threshold ────────────────────────────────────────
KEYSTROKE_THRESHOLD = 0.40   # Phase 4 basic threshold
VOICE_THRESHOLD     = 0.50

# ── Helper: cosine similarity ─────────────────────────────
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0

# ── Helper: feature similarity ────────────────────────────
def feature_similarity(enrolled, live, tolerance=0.30):
    """Compare two feature arrays with tolerance"""
    if not enrolled or not live or len(enrolled) == 0 or len(live) == 0:
        return 0.0  # No data to compare
    
    enrolled = np.array(enrolled)
    live = np.array(live)
    
    min_len = min(len(enrolled), len(live))
    enrolled = enrolled[:min_len]
    live = live[:min_len]

    # Avoid division by zero
    enrolled_safe = np.where(np.abs(enrolled) < 1e-6, 1e-6, enrolled)
    
    # Normalize difference
    diffs = np.abs(enrolled - live) / (np.abs(enrolled_safe) + 1e-6)
    
    # Handle NaN/Inf
    diffs = np.nan_to_num(diffs, nan=1.0, posinf=1.0, neginf=1.0)
    
    score = float(np.mean(diffs < tolerance))
    
    # Ensure valid output
    if np.isnan(score) or np.isinf(score):
        return 0.0
    
    return max(0.0, min(1.0, score))

# ── Log auth attempt ──────────────────────────────────────
def log_attempt(db, user_id, method, confidence, result):
    log = AuthLog(
        user_id=user_id,
        auth_method=method,
        confidence_score=confidence,
        result=result
    )
    db.add(log)
    db.commit()

# ── Endpoints ─────────────────────────────────────────────

@router.post("/keystroke")
def verify_keystroke(payload: KeystrokeAuth, db: Session = Depends(get_db)):
    import pickle
    import os
    from scipy.spatial.distance import cosine as cosine_distance
    
    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.is_flagged:
        raise HTTPException(status_code=403, detail="Account flagged")

    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        '..', 'ml', 'models'
    )
    
    model_path = os.path.join(model_dir, f"{payload.username}_keystroke_rf.pkl")
    
    if os.path.exists(model_path):
        # Use Random Forest + Hybrid Scoring
        print(f"Using Random Forest (Hybrid) for keystroke auth: {payload.username}")
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model = model_data['model']
            scaler = model_data['scaler']
            feature_names = model_data['feature_names']
            enrolled_features = model_data['genuine_features']
            
            # Extract features from current typing
            features = {name: getattr(payload, name, 0.0) for name in feature_names}
            feature_vector = np.array([features[name] for name in feature_names]).reshape(1, -1)
            
            # ═══════════════════════════════════════════════════════════
            # HYBRID SCORING (ML + Template Matching)
            # ═══════════════════════════════════════════════════════════
            
            # 1. Random Forest Prediction
            feature_vector_scaled = scaler.transform(feature_vector)
            proba = model.predict_proba(feature_vector_scaled)[0]
            rf_score = float(proba[1])  # Probability of genuine
            
            # 2. Template Similarity (cosine similarity)
            enrolled_vector = np.array([enrolled_features[name] for name in feature_names])
            live_vector = feature_vector[0]
            
            # Handle edge cases
            if np.linalg.norm(enrolled_vector) > 0 and np.linalg.norm(live_vector) > 0:
                similarity = 1 - cosine_distance(enrolled_vector, live_vector)
                similarity = max(0, min(1, similarity))  # Clamp to [0, 1]
            else:
                similarity = 0.0
            
            # 3. Combine scores (60% RF, 40% similarity)
            confidence = 0.60 * rf_score + 0.40 * similarity
            
            print(f"  RF score: {rf_score:.3f}, Similarity: {similarity:.3f}")
            print(f"  Combined confidence: {confidence:.3f}")
            
            # ═══════════════════════════════════════════════════════════
            # ADAPTIVE THRESHOLD (Progressive Learning - Design 3)
            # ═══════════════════════════════════════════════════════════
            
            login_count = db.query(AuthLog).filter(
                AuthLog.user_id == user.id,
                AuthLog.result == "granted",
                AuthLog.auth_method == "keystroke"
            ).count()
            
            if login_count < 5:
                threshold = 0.35  # Early phase - lenient
                phase = "early"
            elif login_count < 15:
                threshold = 0.45  # Growth phase
                phase = "growth"
            else:
                threshold = 0.55  # Mature phase - strict
                phase = "mature"
            
            authenticated = confidence >= threshold
            
            print(f"  Phase: {phase}, Threshold: {threshold}, Logins: {login_count}")
            print(f"  Result: {'PASS' if authenticated else 'FAIL'}")
            
        except Exception as e:
            print(f"Hybrid scoring error: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback
            template = db.query(KeystrokeTemplate).filter(
                KeystrokeTemplate.user_id == user.id
            ).first()
            if not template:
                raise HTTPException(status_code=404, detail="No keystroke template")
            
            confidence = 0.5
            authenticated = False
    
    else:
        # Simple matching fallback
        print(f"No RF model, using simple matching: {payload.username}")
        
        template = db.query(KeystrokeTemplate).filter(
            KeystrokeTemplate.user_id == user.id
        ).first()
        if not template:
            raise HTTPException(status_code=404, detail="No keystroke template")
        
        dwell_score = feature_similarity(
            template.dwell_times, payload.dwell_times, tolerance=0.35
        )
        confidence = float(dwell_score)
        authenticated = confidence >= 0.40

    print(f"Keystroke auth — user: {payload.username}")
    print(f"  Final confidence: {confidence:.3f}, Result: {'PASS' if authenticated else 'FAIL'}")

    log_attempt(
        db, user.id, "keystroke",
        float(confidence),
        "granted" if authenticated else "denied"
    )

    if not authenticated:
        recent_fails = db.query(AuthLog).filter(
            AuthLog.user_id == user.id,
            AuthLog.result == "denied"
        ).count()

        if recent_fails >= 5:
            user.is_flagged = True
            db.commit()

    return {
        "authenticated": authenticated,
        "confidence": float(confidence)
    }


@router.post("/voice")
def verify_voice(payload: VoiceAuth, db: Session = Depends(get_db)):
    import pickle
    import os
    
    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Check if CNN model exists
    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        '..', 'ml', 'models'
    )
    
    # Try both TensorFlow (.h5) and PyTorch (.pth) models
    tf_model_path = os.path.join(model_dir, f"{payload.username}_voice_cnn.h5")
    pytorch_model_path = os.path.join(model_dir, f"{payload.username}_voice_cnn.pth")
    metadata_path = os.path.join(model_dir, f"{payload.username}_voice_metadata.pkl")
    
    print(f"Looking for model at: {pytorch_model_path}")
    print(f"Model exists: {os.path.exists(pytorch_model_path)}")
    
    if os.path.exists(pytorch_model_path) and os.path.exists(metadata_path):
        # Use PyTorch CNN model
        print(f"Using PyTorch CNN model for voice auth: {payload.username}")
        
        try:
            import torch
            import torch.nn as nn
            
            # Define the same model architecture
            class VoiceCNN_PyTorch(nn.Module):
                def __init__(self):
                    super(VoiceCNN_PyTorch, self).__init__()
                    self.fc1 = nn.Linear(13, 64)
                    self.dropout1 = nn.Dropout(0.3)
                    self.fc2 = nn.Linear(64, 32)
                    self.dropout2 = nn.Dropout(0.3)
                    self.fc3 = nn.Linear(32, 16)
                    self.fc4 = nn.Linear(16, 1)
                    self.sigmoid = nn.Sigmoid()
                
                def forward(self, x):
                    x = torch.relu(self.fc1(x))
                    x = self.dropout1(x)
                    x = torch.relu(self.fc2(x))
                    x = self.dropout2(x)
                    x = torch.relu(self.fc3(x))
                    x = self.fc4(x)
                    x = self.sigmoid(x)
                    return x
            
            # Load model
            model = VoiceCNN_PyTorch()
            model.load_state_dict(torch.load(pytorch_model_path))
            model.eval()
            
            # Predict
            features = torch.FloatTensor(payload.mfcc_features).unsqueeze(0)
            with torch.no_grad():
                prediction = model(features).item()
            
            confidence = float(prediction)
            authenticated = confidence >= 0.50
            
            print(f"CNN prediction: {confidence:.3f}")
            
        except Exception as e:
            print(f"PyTorch CNN model error: {e}, falling back to cosine similarity")
            import traceback
            traceback.print_exc()
            
            # Fallback to cosine similarity
            template = db.query(VoiceTemplate).filter(
                VoiceTemplate.user_id == user.id
            ).first()
            if not template:
                raise HTTPException(status_code=404, detail="No voice template")
            
            confidence = cosine_similarity(
                template.mfcc_features,
                payload.mfcc_features
            )
            confidence = (confidence + 1) / 2
            authenticated = confidence >= VOICE_THRESHOLD
    
    elif os.path.exists(tf_model_path) and os.path.exists(metadata_path):
        # Use TensorFlow CNN model
        print(f"Using TensorFlow CNN model for voice auth: {payload.username}")
        
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(tf_model_path)
            
            # Predict
            features = np.array(payload.mfcc_features).reshape(1, -1)
            prediction = model.predict(features, verbose=0)[0][0]
            
            confidence = float(prediction)
            authenticated = confidence >= 0.70
            
        except Exception as e:
            print(f"TensorFlow CNN model error: {e}, falling back to cosine similarity")
            
            template = db.query(VoiceTemplate).filter(
                VoiceTemplate.user_id == user.id
            ).first()
            if not template:
                raise HTTPException(status_code=404, detail="No voice template")
            
            confidence = cosine_similarity(
                template.mfcc_features,
                payload.mfcc_features
            )
            confidence = (confidence + 1) / 2
            authenticated = confidence >= VOICE_THRESHOLD
    
    else:
        # No model exists, use simple cosine similarity
        print(f"No CNN model found, using cosine similarity: {payload.username}")
        
        template = db.query(VoiceTemplate).filter(
            VoiceTemplate.user_id == user.id
        ).first()
        if not template:
            raise HTTPException(status_code=404, detail="No voice template found")
        
        confidence = cosine_similarity(
            template.mfcc_features,
            payload.mfcc_features
        )
        confidence = (confidence + 1) / 2
        authenticated = confidence >= VOICE_THRESHOLD

    print(f"Voice auth — user: {payload.username}")
    print(f"  confidence: {confidence:.3f}, result: {'PASS' if authenticated else 'FAIL'}")

    log_attempt(
        db, user.id, "voice",
        confidence,
        "granted" if authenticated else "denied"
    )

    return {
        "authenticated": authenticated,
        "confidence": confidence
    }


@router.get("/security-question/{username}")
def get_security_question(username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    sq = db.query(SecurityQuestion).filter(
        SecurityQuestion.user_id == user.id
    ).first()
    if not sq:
        raise HTTPException(status_code=404, detail="No security question found")

    return {"question": sq.question}


@router.post("/security")
def verify_security(payload: SecurityAuth, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    sq = db.query(SecurityQuestion).filter(
        SecurityQuestion.user_id == user.id
    ).first()
    if not sq:
        raise HTTPException(status_code=404, detail="No security question found")

    # Hash the provided answer
    answer_hash = hashlib.sha256(
        payload.answer.strip().lower().encode()
    ).hexdigest()

    authenticated = answer_hash == sq.answer_hash

    print(f"Security question auth — user: {payload.username}")
    print(f"  result: {'PASS' if authenticated else 'FAIL'}")

    log_attempt(
        db, user.id, "security_question",
        1.0 if authenticated else 0.0,
        "granted" if authenticated else "denied"
    )

    # Flag user if security question also fails
    if not authenticated:
        user.is_flagged = True
        db.commit()
        print(f"  ⚠️ User {payload.username} flagged!")

    return {
        "authenticated": authenticated,
        "confidence": 1.0 if authenticated else 0.0
    }