import redis
import json
from app.models import SessionData, Message
from app.config import Config

redis_client = redis.from_url(Config.REDIS_URL, decode_responses=True)

def save_session(session_id: str, session_data: SessionData):
    key = f"session:{session_id}"
    redis_client.set(key, session_data.json())

def load_session(session_id: str) -> SessionData:
    key = f"session:{session_id}"
    data = redis_client.get(key)
    if data:
        return SessionData.parse_raw(data)
    return SessionData(session_id=session_id, history=[])

def delete_session(session_id: str):
    redis_client.delete(f"session:{session_id}")