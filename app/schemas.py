from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator


class SeriesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=80)
    values: list[FiniteFloat] = Field(min_length=32, max_length=1024)

    @field_validator("values", mode="before")
    @classmethod
    def reject_booleans(cls, values):
        if isinstance(values, list) and any(isinstance(v, bool) for v in values):
            raise ValueError("Values must be numbers, not booleans")
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


class Candle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp_ms: int = Field(
        ge=0, le=253402300799999, strict=True, description="Opening time in UTC milliseconds"
    )
    close: FiniteFloat

    @field_validator("close", mode="before")
    @classmethod
    def reject_boolean(cls, value):
        if isinstance(value, bool):
            raise ValueError("Close must be a number, not a boolean")
        return value


class CandleForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=80)
    candles: list[Candle] = Field(min_length=32, max_length=1024)
    interval_ms: int = Field(ge=1, le=604800000, strict=True)
    context: int | None = Field(default=None, ge=32, le=1024, strict=True)
    horizon: int = Field(default=4, ge=1, le=256, strict=True)
    return_quantiles: bool = True
    require_fresh: bool = False


class MarketForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str = Field(min_length=1, max_length=40)
    symbol: str = Field(min_length=1, max_length=80)
    interval: str = Field(default="1m", min_length=1, max_length=10)
    context: int = Field(default=128, ge=32, le=1024, strict=True)
    horizon: int = Field(default=4, ge=1, le=256, strict=True)
    return_quantiles: bool = True


class CandleForecastResponse(BaseModel):
    id: str
    context_end: str
    forecast_close_times: list[str]
    prediction: ForecastResponse


class MarketForecastResponse(BaseModel):
    provider: str
    symbol: str
    interval: str
    context_end: str
    data_fetch_ms: float
    forecast_close_times: list[str]
    prediction: ForecastResponse
