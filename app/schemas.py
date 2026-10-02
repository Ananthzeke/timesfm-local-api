from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator


class SeriesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=80)
    values: list[FiniteFloat] = Field(min_length=32, max_length=1024)

    @field_validator("values", mode="before")
    @classmethod
    def reject_booleans(cls, values):
        if isinstance(values, list) and any(isinstance(v, bool) for v in values):
            raise ValueError("Prices must be numbers, not booleans")
        return values


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    series: list[SeriesInput] = Field(min_length=1, max_length=8)
    horizon: int = Field(default=4, ge=1, le=256, strict=True)
    return_quantiles: bool = True

    @field_validator("series")
    @classmethod
    def unique_ids(cls, series):
        if len({s.id for s in series}) != len(series):
            raise ValueError("Series IDs must be unique within a request")
        return series


class SeriesForecast(BaseModel):
    id: str
    forecast: list[FiniteFloat]
    quantiles: dict[str, list[FiniteFloat]] | None = None


class Timing(BaseModel):
    queue_ms: float
    inference_batch_ms: float
    service_ms: float
    batch_series: int


class ForecastResponse(BaseModel):
    request_id: str
    backend: str
    model: str
    device: str
    horizon: int
    results: list[SeriesForecast]
    timing: Timing
