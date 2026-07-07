import datetime
from typing import Optional, Union, Any
from jose import jwt, JWTError
from passlib.context import CryptContext
from src.core.config import settings

# Configure specialized passlib context to use highly secure bcrypt hashing schemes
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Use standardized cryptographic algorithm parameters for JWT signatures
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

def hash_password(password: str) -> str:
    """Generates a secure cryptographic salt and hashes the raw plain password string."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password string matches its computed cryptographic signature record."""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None) -> str:
    """Compiles a signed JSON Web Token containing distinct user identity payload metadata grids."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt