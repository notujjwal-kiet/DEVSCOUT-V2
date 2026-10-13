"""DEVSCOUT backend – application entry point."""
import uvicorn

from app.config import get_settings

if __name__ == "__main__":
    cfg = get_settings()
    uvicorn.run(
        "app.main:app",
        host=cfg.app_host,
        port=cfg.app_port,
        reload=(cfg.app_env == "development"),
        log_level=cfg.log_level.lower(),
    )
