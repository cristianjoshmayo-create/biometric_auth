# backend/utils/crypto.py
# AES symmetric encryption for sensitive database fields.
#
# Uses Fernet (from the cryptography library) which provides:
#   - AES-128-CBC encryption with PKCS7 padding
#   - HMAC-SHA256 authentication (prevents tampering / bit-flipping)
#   - Timestamp embedded in the token (allows key expiry if needed)
#
# Fields encrypted with this module:
#   - users.phrase              — passphrase the user must speak at login
#   - security_questions.question — reveals personal info if exposed
#
# The ENCRYPTION_KEY must live in .env — never hardcode it in source code.
#
# One-time setup:
#   1. Generate a key:
#        python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
#   2. Add to .env:
#        ENCRYPTION_KEY=<paste key here>
#   3. Keep the key backed up — data encrypted with a lost key cannot be recovered.

import os
from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv

load_dotenv()

_fernet_instance = None


def _get_fernet() -> Fernet:
    global _fernet_instance
    if _fernet_instance is not None:
        return _fernet_instance
    key = os.getenv("ENCRYPTION_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "ENCRYPTION_KEY is not set in .env. "
            "Generate one with:\n"
            "  python -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\"\n"
            "Then add  ENCRYPTION_KEY=<value>  to your .env file."
        )
    _fernet_instance = Fernet(key.encode())
    return _fernet_instance


def encrypt(plaintext: str) -> str:
    """
    Encrypt a plaintext string.
    Returns a URL-safe base64 Fernet token (stores safely in any text column).
    Returns empty string if plaintext is empty.
    """
    if not plaintext:
        return ""
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt(token: str) -> str:
    """
    Decrypt a Fernet token back to the original plaintext.
    Returns empty string if token is empty.
    Falls back to returning the raw value if decryption fails — this protects
    against a hard crash if any legacy unencrypted value is still in the DB
    during a transition period.
    """
    if not token:
        return ""
    try:
        return _get_fernet().decrypt(token.encode()).decode()
    except (InvalidToken, Exception):
        return token
