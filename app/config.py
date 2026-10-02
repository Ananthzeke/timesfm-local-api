from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TF_", env_file=".env", extra="ignore")

    backend: Literal["timesfm3", "mock"] = "timesfm3"
    device: Literal["cuda", "cpu"] = "cuda"
    checkpoint: str = "google/timesfm-3.0-pytorch"
    model_revision: str | None = None
    license_accepted: bool = False
    max_context: int = Field(default=512, ge=32, le=1024)
    max_horizon: int = Field(default=64, ge=1, le=256)
    max_batch_series: int = Field(default=1, ge=1, le=8)
    batch_window_ms: float = Field(default=0, ge=0, le=20)
    queue_capacity: int = Field(default=32, ge=1, le=256)
    request_timeout_seconds: float = Field(default=15, gt=0, le=300)
    warmup_runs: int = Field(default=2, ge=1, le=10)
    gc_interval_seconds: float = Field(default=30, ge=0, le=300)
    max_body_bytes: int = Field(default=131072, ge=4096, le=1048576)
    upstream_timeout_seconds: float = Field(default=5, gt=0, le=30)
    max_upstream_requests: int = Field(default=4, ge=1, le=32)
    api_key: SecretStr | None = None
    api_url: AnyHttpUrl = "http://127.0.0.1:8000"

    @model_validator(mode="after")
    def validate_key(self):
        if self.api_key is not None and not self.api_key.get_secret_value():
            self.api_key = None
        return self
