import os, requests
from ..settings import ZAPIER_WEBHOOK_URL

def fire_event(event: str, payload: dict):
    if not ZAPIER_WEBHOOK_URL:
        return
    try:
        requests.post(ZAPIER_WEBHOOK_URL, json={"event": event, **payload}, timeout=4)
    except requests.RequestException:
        pass   # fail-silent — don’t break user flow
