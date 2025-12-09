"""
Security utilities for Hypothesis Forge.
Handles sensitive data encryption and secure configuration.
"""
import os
import hashlib
from typing import Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64


class SecureConfig:
    """Handles secure configuration management."""

    @staticmethod
    def generate_key(password: str, salt: Optional[bytes] = None) -> bytes:
        """
        Generate encryption key from password.

        Args:
            password: Password string
            salt: Optional salt bytes

        Returns:
            Encryption key
        """
        if salt is None:
            salt = os.urandom(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    @staticmethod
    def encrypt_value(value: str, key: bytes) -> str:
        """
        Encrypt a value.

        Args:
            value: Value to encrypt
            key: Encryption key

        Returns:
            Encrypted value (base64 encoded)
        """
        f = Fernet(key)
        encrypted = f.encrypt(value.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    @staticmethod
    def decrypt_value(encrypted_value: str, key: bytes) -> str:
        """
        Decrypt a value.

        Args:
            encrypted_value: Encrypted value (base64 encoded)
            key: Encryption key

        Returns:
            Decrypted value
        """
        f = Fernet(key)
        decoded = base64.urlsafe_b64decode(encrypted_value.encode())
        decrypted = f.decrypt(decoded)
        return decrypted.decode()

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """
        Hash a password securely.

        Args:
            password: Password to hash
            salt: Optional salt string

        Returns:
            Tuple of (hashed_password, salt)
        """
        if salt is None:
            salt = os.urandom(16).hex()

        # Use SHA-256 for hashing (in production, use bcrypt or argon2)
        hash_obj = hashlib.sha256()
        hash_obj.update((password + salt).encode())
        hashed = hash_obj.hexdigest()

        return hashed, salt


def validate_api_key(api_key: Optional[str]) -> bool:
    """
    Validate API key format.

    Args:
        api_key: API key to validate

    Returns:
        True if valid format, False otherwise
    """
    if not api_key:
        return False
    # Basic validation - check length and format
    return len(api_key) >= 8 and isinstance(api_key, str)


def sanitize_input(user_input: str, max_length: int = 1000) -> str:
    """
    Sanitize user input to prevent injection attacks.

    Args:
        user_input: User input string
        max_length: Maximum allowed length

    Returns:
        Sanitized input
    """
    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '"', "'", '&', ';', '|', '`', '$']
    sanitized = user_input
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')

    # Truncate to max length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    return sanitized.strip()

