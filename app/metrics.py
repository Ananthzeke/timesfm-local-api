from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        reg = self.registry
        self.http_seconds = Histogram(
            "tf_http_request_seconds",
            "HTTP handling including serialization",
            ["route", "status"],
            registry=reg,
        )
        self.accepted = Counter("tf_accepted_requests", "Requests admitted", registry=reg)
        self.rejected = Counter(
            "tf_rejected_requests",
            "Requests rejected",
            ["reason"],
            registry=reg,
        )
        self.queue_seconds = Histogram(
            "tf_queue_seconds",
            "Time before dispatch, including batching window",
            registry=reg,
        )
        self.inference_seconds = Histogram(
            "tf_inference_batch_seconds",
            "Synchronized end-to-end inference batch time",
            registry=reg,
        )
        self.batch_size = Histogram(
            "tf_batch_series",
            "Series per dispatch",
            buckets=(1, 2, 4, 8),
            registry=reg,
        )
        self.completed_series = Counter(
            "tf_completed_series",
            "Successfully inferred series",
            registry=reg,
        )
        self.inference_errors = Counter("tf_inference_errors", "Failed batches", registry=reg)
        self.oom_retries = Counter("tf_oom_retries", "Batch retries after GPU OOM", registry=reg)
        self.gc_collections = Counter(
            "tf_gc_collections", "Explicit model cleanup collections", ["reason"], registry=reg
        )
        self.outstanding = Gauge(
            "tf_outstanding_requests",
            "Queued and running requests",
            registry=reg,
        )
        self.queue_depth = Gauge("tf_queue_depth", "Queued requests", registry=reg)
        self.gpu_bytes = Gauge(
            "tf_gpu_memory_bytes",
            "PyTorch allocated/reserved/peak bytes",
            ["kind"],
            registry=reg,
        )
