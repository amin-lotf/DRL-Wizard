import os
import uvicorn
from .app import app

def main():
    host = os.getenv("DRL_API_HOST", "0.0.0.0")
    port = int(os.getenv("DRL_API_PORT", "8000"))
    reload = os.getenv("DRL_API_RELOAD", "false").lower() == "true"
    uvicorn.run(app, host=host, port=port, reload=reload)
