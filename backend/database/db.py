# backend/database/db.py

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST     = os.getenv("DB_HOST")
DB_PORT     = os.getenv("DB_PORT")
DB_NAME     = os.getenv("DB_NAME")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ── Supabase connection pool settings ────────────────────────────────────────
# Supabase closes idle connections after ~60 seconds.
# ML training takes 30–120 seconds — the connection goes stale mid-run.
# pool_pre_ping retests the connection before each use and auto-reconnects.
# pool_recycle replaces connections every 55s before Supabase drops them.
# keepalives send TCP-level heartbeats so the connection stays alive.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=55,
    pool_size=5,
    max_overflow=10,
    connect_args={
        "keepalives":          1,
        "keepalives_idle":     30,
        "keepalives_interval": 10,
        "keepalives_count":    5,
        "connect_timeout":     10,
    }
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency — inject this into your FastAPI routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()