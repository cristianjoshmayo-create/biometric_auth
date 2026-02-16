# backend/database/models.py

from sqlalchemy import (
    Column, Integer, String, Float, 
    ARRAY, ForeignKey, DateTime, Text, Boolean
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database.db import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False)
    is_flagged = Column(Boolean, default=False)  # attacker flag
    created_at = Column(DateTime, default=func.now())

    # Relationships
    keystroke_template = relationship("KeystrokeTemplate", back_populates="user")
    voice_template = relationship("VoiceTemplate", back_populates="user")
    security_question = relationship("SecurityQuestion", back_populates="user")
    auth_logs = relationship("AuthLog", back_populates="user")


class KeystrokeTemplate(Base):
    __tablename__ = "keystroke_templates"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    dwell_times = Column(ARRAY(Float), nullable=False)
    flight_times = Column(ARRAY(Float), nullable=False)
    typing_speed = Column(Float)
    enrolled_at = Column(DateTime, default=func.now())

    user = relationship("User", back_populates="keystroke_template")


class VoiceTemplate(Base):
    __tablename__ = "voice_templates"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    mfcc_features = Column(ARRAY(Float), nullable=False)
    enrolled_at = Column(DateTime, default=func.now())

    user = relationship("User", back_populates="voice_template")


class SecurityQuestion(Base):
    __tablename__ = "security_questions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    question = Column(Text, nullable=False)
    answer_hash = Column(Text, nullable=False)  # never store plain text

    user = relationship("User", back_populates="security_question")


class AuthLog(Base):
    __tablename__ = "auth_logs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    auth_method = Column(String(50))    # 'keystroke', 'voice', 'security_question'
    confidence_score = Column(Float)
    result = Column(String(20))         # 'granted', 'denied', 'flagged'
    failed_attempts = Column(Integer, default=0)
    attempted_at = Column(DateTime, default=func.now())

    user = relationship("User", back_populates="auth_logs")