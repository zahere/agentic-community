"""
OpenTelemetry integration for comprehensive observability.

Provides distributed tracing, metrics, and logging for the Agentic Framework.
"""

import time
from typing import Dict, Any, Optional, Callable, List
from functools import wraps
from contextlib import contextmanager
import logging

try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logging.warning("OpenTelemetry not installed. Install with: pip install opentelemetry-distro")

logger = logging.getLogger(__name__)


class ObservabilityConfig:
    """Configuration for observability features."""
    
    def __init__(
        self,
        service_name: str = "agentic-framework",
        otlp_endpoint: str = "localhost:4317",
        enable_tracing: bool = True,
        enable_metrics: bool = True,
        enable_logging: bool = True,
        sample_rate: float = 1.0,
        export_interval_millis: int = 5000
    ):
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self.enable_tracing = enable_tracing
        self.enable_metrics = enable_metrics
        self.enable_logging = enable_logging
        self.sample_rate = sample_rate
        self.export_interval_millis = export_interval_millis


class ObservabilityManager:
    """Manages OpenTelemetry instrumentation for the framework."""
    
    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self.tracer = None
        self.meter = None
        self._initialized = False
        
        # Metrics
        self.request_counter = None
        self.request_duration = None
        self.tool_usage_counter = None
        self.llm_token_counter = None
        self.error_counter = None
        
        if OTEL_AVAILABLE:
            self._initialize()
            
    def _initialize(self):
        """Initialize OpenTelemetry providers."""
        if self._initialized:
            return
            
        # Create resource
        resource = Resource.create({
            "service.name": self.config.service_name,
            "service.version": "1.0.0",
        })
        
        # Initialize tracing
        if self.config.enable_tracing:
            # Set up tracer provider
            tracer_provider = TracerProvider(resource=resource)
            
            # Add OTLP exporter
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.config.otlp_endpoint,
                insecure=True
            )
            span_processor = BatchSpanProcessor(otlp_exporter)
            tracer_provider.add_span_processor(span_processor)
            
            # Set global tracer provider
            trace.set_tracer_provider(tracer_provider)
            self.tracer = trace.get_tracer(__name__)
            
        # Initialize metrics
        if self.config.enable_metrics:
            # Set up meter provider
            metric_exporter = OTLPMetricExporter(
                endpoint=self.config.otlp_endpoint,
                insecure=True
            )
            metric_reader = PeriodicExportingMetricReader(
                exporter=metric_exporter,
                export_interval_millis=self.config.export_interval_millis
            )
            meter_provider = MeterProvider(
                resource=resource,
                metric_readers=[metric_reader]
            )
            metrics.set_meter_provider(meter_provider)
            self.meter = metrics.get_meter(__name__)
            
            # Create metrics
            self._create_metrics()
            
        self._initialized = True
        logger.info("OpenTelemetry initialized successfully")
        
    def _create_metrics(self):
        """Create metric instruments."""
        if not self.meter:
            return
            
        self.request_counter = self.meter.create_counter(
            name="agentic.requests.total",
            description="Total number of requests",
            unit="1"
        )
        
        self.request_duration = self.meter.create_histogram(
            name="agentic.request.duration",
            description="Request duration in milliseconds",
            unit="ms"
        )
        
        self.tool_usage_counter = self.meter.create_counter(
            name="agentic.tool.usage",
            description="Tool usage count",
            unit="1"
        )
        
        self.llm_token_counter = self.meter.create_counter(
            name="agentic.llm.tokens",
            description="LLM token usage",
            unit="1"
        )
        
        self.error_counter = self.meter.create_counter(
            name="agentic.errors.total",
            description="Total number of errors",
            unit="1"
        )
        
    @contextmanager
    def trace_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create a trace span."""
        if not self.tracer:
            yield None
            return
            
        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span
            
    def record_request(self, endpoint: str, method: str, duration_ms: float, status_code: int):
        """Record HTTP request metrics."""
        if not self.request_counter:
            return
            
        labels = {
            "endpoint": endpoint,
            "method": method,
            "status_code": str(status_code)
        }
        
        self.request_counter.add(1, labels)
        self.request_duration.record(duration_ms, labels)
        
    def record_tool_usage(self, tool_name: str, success: bool, duration_ms: float):
        """Record tool usage metrics."""
        if not self.tool_usage_counter:
            return
            
        labels = {
            "tool": tool_name,
            "success": str(success)
        }
        
        self.tool_usage_counter.add(1, labels)
        
    def record_llm_tokens(self, provider: str, model: str, prompt_tokens: int, completion_tokens: int):
        """Record LLM token usage."""
        if not self.llm_token_counter:
            return
            
        labels = {
            "provider": provider,
            "model": model,
            "type": "prompt"
        }
        self.llm_token_counter.add(prompt_tokens, labels)
        
        labels["type"] = "completion"
        self.llm_token_counter.add(completion_tokens, labels)
        
    def record_error(self, error_type: str, component: str):
        """Record error occurrence."""
        if not self.error_counter:
            return
            
        labels = {
            "error_type": error_type,
            "component": component
        }
        
        self.error_counter.add(1, labels)
        
    def instrument_fastapi(self, app):
        """Instrument FastAPI application."""
        if OTEL_AVAILABLE and self.config.enable_tracing:
            FastAPIInstrumentor.instrument_app(app)
            
    def instrument_httpx(self):
        """Instrument HTTPX client."""
        if OTEL_AVAILABLE and self.config.enable_tracing:
            HTTPXClientInstrumentor().instrument()


# Global observability manager
_observability_manager: Optional[ObservabilityManager] = None


def initialize_observability(config: Optional[ObservabilityConfig] = None):
    """Initialize global observability manager."""
    global _observability_manager
    
    if not config:
        config = ObservabilityConfig()
        
    _observability_manager = ObservabilityManager(config)
    return _observability_manager


def get_observability_manager() -> Optional[ObservabilityManager]:
    """Get the global observability manager."""
    return _observability_manager


# Decorators for instrumentation
def trace_method(name: Optional[str] = None):
    """Decorator to trace method execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_observability_manager()
            if not manager or not manager.tracer:
                return func(*args, **kwargs)
                
            span_name = name or f"{func.__module__}.{func.__name__}"
            
            with manager.trace_span(span_name) as span:
                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    duration = (time.time() - start_time) * 1000
                    
                    if span:
                        span.set_attribute("duration_ms", duration)
                        span.set_status(trace.Status(trace.StatusCode.OK))
                        
                    return result
                except Exception as e:
                    if span:
                        span.set_status(
                            trace.Status(trace.StatusCode.ERROR, str(e))
                        )
                        span.record_exception(e)
                    raise
                    
        return wrapper
    return decorator


def trace_tool(tool_name: str):
    """Decorator to trace tool execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_observability_manager()
            if not manager:
                return func(*args, **kwargs)
                
            start_time = time.time()
            success = True
            
            try:
                with manager.trace_span(f"tool.{tool_name}") as span:
                    if span:
                        span.set_attribute("tool.name", tool_name)
                        
                    result = func(*args, **kwargs)
                    return result
            except Exception as e:
                success = False
                manager.record_error(type(e).__name__, f"tool.{tool_name}")
                raise
            finally:
                duration = (time.time() - start_time) * 1000
                manager.record_tool_usage(tool_name, success, duration)
                
        return wrapper
    return decorator


def trace_llm_call(provider: str, model: str):
    """Decorator to trace LLM calls."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            manager = get_observability_manager()
            if not manager:
                return await func(*args, **kwargs)
                
            with manager.trace_span(f"llm.{provider}.{model}") as span:
                if span:
                    span.set_attribute("llm.provider", provider)
                    span.set_attribute("llm.model", model)
                    
                try:
                    result = await func(*args, **kwargs)
                    
                    # Extract token usage if available
                    if hasattr(result, 'usage'):
                        prompt_tokens = result.usage.get('prompt_tokens', 0)
                        completion_tokens = result.usage.get('completion_tokens', 0)
                        
                        manager.record_llm_tokens(
                            provider, model, 
                            prompt_tokens, completion_tokens
                        )
                        
                        if span:
                            span.set_attribute("llm.prompt_tokens", prompt_tokens)
                            span.set_attribute("llm.completion_tokens", completion_tokens)
                            
                    return result
                except Exception as e:
                    manager.record_error(type(e).__name__, f"llm.{provider}")
                    raise
                    
        return wrapper
    return decorator


# Middleware for API observability
class ObservabilityMiddleware:
    """Middleware for API observability."""
    
    def __init__(self, app):
        self.app = app
        
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
            
        manager = get_observability_manager()
        if not manager:
            await self.app(scope, receive, send)
            return
            
        start_time = time.time()
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                duration = (time.time() - start_time) * 1000
                
                manager.record_request(
                    endpoint=scope["path"],
                    method=scope["method"],
                    duration_ms=duration,
                    status_code=message["status"]
                )
                
            await send(message)
            
        await self.app(scope, receive, send_wrapper)
