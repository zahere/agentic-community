"""
OpenTelemetry Integration for Agentic Framework

This module provides OpenTelemetry instrumentation for distributed tracing,
metrics, and logging across the entire framework.
"""

import os
from typing import Dict, Any, Optional, Callable, List
from functools import wraps
import asyncio
from contextlib import contextmanager

# OpenTelemetry imports
from opentelemetry import trace, metrics, baggage
from opentelemetry.context import attach, detach
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.metrics import Counter, Histogram, UpDownCounter

from agentic_community.core.exceptions import ObservabilityError


class TelemetryConfig:
    """Configuration for OpenTelemetry"""
    
    def __init__(
        self,
        service_name: str = "agentic-framework",
        service_version: str = "1.0.0",
        otlp_endpoint: Optional[str] = None,
        prometheus_port: int = 9090,
        enable_console_export: bool = False,
        enable_logging: bool = True,
        trace_sample_rate: float = 1.0,
        metric_export_interval: int = 60000,  # milliseconds
        custom_attributes: Optional[Dict[str, Any]] = None
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.otlp_endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        self.prometheus_port = prometheus_port
        self.enable_console_export = enable_console_export
        self.enable_logging = enable_logging
        self.trace_sample_rate = trace_sample_rate
        self.metric_export_interval = metric_export_interval
        self.custom_attributes = custom_attributes or {}
        
        # Resource attributes
        self.resource = Resource.create({
            SERVICE_NAME: self.service_name,
            SERVICE_VERSION: self.service_version,
            "deployment.environment": os.getenv("ENVIRONMENT", "development"),
            **self.custom_attributes
        })


class OpenTelemetryIntegration:
    """
    OpenTelemetry integration for the Agentic Framework
    
    Features:
    - Distributed tracing with context propagation
    - Metrics collection (counters, histograms, gauges)
    - Automatic instrumentation of agents and tools
    - Custom span attributes and events
    - Baggage propagation for metadata
    - Multiple export formats (OTLP, Prometheus, Console)
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, config: Optional[TelemetryConfig] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[TelemetryConfig] = None):
        if not self._initialized:
            self.config = config or TelemetryConfig()
            self._setup_tracing()
            self._setup_metrics()
            self._setup_logging()
            self._setup_propagators()
            self._initialized = True
    
    def _setup_tracing(self):
        """Set up tracing provider and exporters"""
        # Create tracer provider
        tracer_provider = TracerProvider(
            resource=self.config.resource,
            sampler=trace.sampling.TraceIdRatioBased(self.config.trace_sample_rate)
        )
        
        # Add exporters
        if self.config.otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.config.otlp_endpoint,
                insecure=True  # Use secure=False for development
            )
            tracer_provider.add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )
        
        if self.config.enable_console_export:
            console_exporter = ConsoleSpanExporter()
            tracer_provider.add_span_processor(
                BatchSpanProcessor(console_exporter)
            )
        
        # Set global tracer provider
        trace.set_tracer_provider(tracer_provider)
        
        # Create tracer
        self.tracer = trace.get_tracer(
            self.config.service_name,
            self.config.service_version
        )
    
    def _setup_metrics(self):
        """Set up metrics provider and exporters"""
        # Create metric readers
        metric_readers = []
        
        if self.config.otlp_endpoint:
            otlp_reader = PeriodicExportingMetricReader(
                exporter=OTLPMetricExporter(
                    endpoint=self.config.otlp_endpoint,
                    insecure=True
                ),
                export_interval_millis=self.config.metric_export_interval
            )
            metric_readers.append(otlp_reader)
        
        # Add Prometheus exporter
        prometheus_reader = PrometheusMetricReader()
        metric_readers.append(prometheus_reader)
        
        # Create meter provider
        meter_provider = MeterProvider(
            resource=self.config.resource,
            metric_readers=metric_readers
        )
        
        # Set global meter provider
        metrics.set_meter_provider(meter_provider)
        
        # Create meter
        self.meter = metrics.get_meter(
            self.config.service_name,
            self.config.service_version
        )
        
        # Create common metrics
        self._create_common_metrics()
    
    def _setup_logging(self):
        """Set up logging instrumentation"""
        if self.config.enable_logging:
            LoggingInstrumentor().instrument()
    
    def _setup_propagators(self):
        """Set up context propagators"""
        # Use B3 propagation for compatibility
        set_global_textmap(B3MultiFormat())
    
    def _create_common_metrics(self):
        """Create commonly used metrics"""
        # Agent metrics
        self.agent_invocations = self.meter.create_counter(
            name="agent_invocations_total",
            description="Total number of agent invocations",
            unit="1"
        )
        
        self.agent_duration = self.meter.create_histogram(
            name="agent_duration_seconds",
            description="Duration of agent execution",
            unit="s"
        )
        
        self.agent_errors = self.meter.create_counter(
            name="agent_errors_total",
            description="Total number of agent errors",
            unit="1"
        )
        
        # Tool metrics
        self.tool_invocations = self.meter.create_counter(
            name="tool_invocations_total",
            description="Total number of tool invocations",
            unit="1"
        )
        
        self.tool_duration = self.meter.create_histogram(
            name="tool_duration_seconds",
            description="Duration of tool execution",
            unit="s"
        )
        
        # LLM metrics
        self.llm_requests = self.meter.create_counter(
            name="llm_requests_total",
            description="Total number of LLM requests",
            unit="1"
        )
        
        self.llm_tokens = self.meter.create_counter(
            name="llm_tokens_total",
            description="Total number of tokens processed",
            unit="1"
        )
        
        self.llm_latency = self.meter.create_histogram(
            name="llm_latency_seconds",
            description="LLM request latency",
            unit="s"
        )
        
        # System metrics
        self.active_agents = self.meter.create_up_down_counter(
            name="active_agents",
            description="Number of active agents",
            unit="1"
        )
        
        self.memory_usage = self.meter.create_observable_gauge(
            name="memory_usage_bytes",
            callbacks=[self._get_memory_usage],
            description="Memory usage in bytes",
            unit="By"
        )
    
    def _get_memory_usage(self, options):
        """Callback for memory usage metric"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss
    
    # Decorator for tracing
    def trace_method(
        self,
        span_name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True
    ):
        """Decorator to trace a method or function"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                name = span_name or f"{func.__module__}.{func.__name__}"
                
                with self.tracer.start_as_current_span(
                    name,
                    attributes=attributes,
                    record_exception=record_exception,
                    set_status_on_exception=set_status_on_exception
                ) as span:
                    try:
                        # Add function arguments as span attributes
                        span.set_attribute("function.args", str(args))
                        span.set_attribute("function.kwargs", str(kwargs))
                        
                        result = await func(*args, **kwargs)
                        
                        span.set_status(Status(StatusCode.OK))
                        return result
                        
                    except Exception as e:
                        if record_exception:
                            span.record_exception(e)
                        if set_status_on_exception:
                            span.set_status(
                                Status(StatusCode.ERROR, str(e))
                            )
                        raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                name = span_name or f"{func.__module__}.{func.__name__}"
                
                with self.tracer.start_as_current_span(
                    name,
                    attributes=attributes,
                    record_exception=record_exception,
                    set_status_on_exception=set_status_on_exception
                ) as span:
                    try:
                        result = func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                        
                    except Exception as e:
                        if record_exception:
                            span.record_exception(e)
                        if set_status_on_exception:
                            span.set_status(
                                Status(StatusCode.ERROR, str(e))
                            )
                        raise
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    # Context managers for tracing
    @contextmanager
    def trace_block(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[List[trace.Link]] = None
    ):
        """Context manager for tracing a code block"""
        with self.tracer.start_as_current_span(
            name,
            attributes=attributes,
            links=links
        ) as span:
            yield span
    
    # Agent instrumentation
    def instrument_agent(self, agent_class):
        """Instrument an agent class for automatic tracing"""
        original_process = agent_class.process
        original_think = getattr(agent_class, 'think', None)
        
        @wraps(original_process)
        async def traced_process(self, *args, **kwargs):
            agent_name = getattr(self, 'name', 'unknown')
            
            with telemetry.trace_block(
                f"agent.process.{agent_name}",
                attributes={
                    "agent.name": agent_name,
                    "agent.type": self.__class__.__name__
                }
            ) as span:
                # Record metric
                telemetry.agent_invocations.add(
                    1,
                    {"agent.name": agent_name}
                )
                
                start_time = asyncio.get_event_loop().time()
                
                try:
                    result = await original_process(self, *args, **kwargs)
                    
                    # Record duration
                    duration = asyncio.get_event_loop().time() - start_time
                    telemetry.agent_duration.record(
                        duration,
                        {"agent.name": agent_name}
                    )
                    
                    return result
                    
                except Exception as e:
                    telemetry.agent_errors.add(
                        1,
                        {"agent.name": agent_name, "error.type": type(e).__name__}
                    )
                    raise
        
        agent_class.process = traced_process
        
        # Also instrument think method if it exists
        if original_think:
            @wraps(original_think)
            async def traced_think(self, *args, **kwargs):
                agent_name = getattr(self, 'name', 'unknown')
                
                with telemetry.trace_block(
                    f"agent.think.{agent_name}",
                    attributes={
                        "agent.name": agent_name,
                        "agent.type": self.__class__.__name__
                    }
                ):
                    return await original_think(self, *args, **kwargs)
            
            agent_class.think = traced_think
        
        return agent_class
    
    # Tool instrumentation
    def instrument_tool(self, tool_class):
        """Instrument a tool class for automatic tracing"""
        original_execute = tool_class.execute
        
        @wraps(original_execute)
        async def traced_execute(self, *args, **kwargs):
            tool_name = getattr(self, 'name', 'unknown')
            
            with telemetry.trace_block(
                f"tool.execute.{tool_name}",
                attributes={
                    "tool.name": tool_name,
                    "tool.type": self.__class__.__name__
                }
            ) as span:
                # Record metric
                telemetry.tool_invocations.add(
                    1,
                    {"tool.name": tool_name}
                )
                
                start_time = asyncio.get_event_loop().time()
                
                try:
                    result = await original_execute(self, *args, **kwargs)
                    
                    # Record duration
                    duration = asyncio.get_event_loop().time() - start_time
                    telemetry.tool_duration.record(
                        duration,
                        {"tool.name": tool_name}
                    )
                    
                    # Add result info to span
                    if hasattr(result, 'success'):
                        span.set_attribute("tool.success", result.success)
                    
                    return result
                    
                except Exception as e:
                    span.set_attribute("tool.error", str(e))
                    raise
        
        tool_class.execute = traced_execute
        return tool_class
    
    # LLM instrumentation
    def trace_llm_call(
        self,
        provider: str,
        model: str,
        messages: List[Dict[str, str]],
        response: Optional[Any] = None,
        error: Optional[Exception] = None,
        duration: Optional[float] = None,
        tokens: Optional[Dict[str, int]] = None
    ):
        """Record LLM call metrics and traces"""
        span = trace.get_current_span()
        
        # Set span attributes
        if span.is_recording():
            span.set_attribute("llm.provider", provider)
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.message_count", len(messages))
            
            if tokens:
                span.set_attribute("llm.prompt_tokens", tokens.get("prompt_tokens", 0))
                span.set_attribute("llm.completion_tokens", tokens.get("completion_tokens", 0))
                span.set_attribute("llm.total_tokens", tokens.get("total_tokens", 0))
            
            if error:
                span.record_exception(error)
                span.set_status(Status(StatusCode.ERROR, str(error)))
        
        # Record metrics
        self.llm_requests.add(
            1,
            {
                "llm.provider": provider,
                "llm.model": model,
                "llm.status": "error" if error else "success"
            }
        )
        
        if tokens:
            self.llm_tokens.add(
                tokens.get("total_tokens", 0),
                {"llm.provider": provider, "llm.model": model}
            )
        
        if duration:
            self.llm_latency.record(
                duration,
                {"llm.provider": provider, "llm.model": model}
            )
    
    # Baggage operations
    def set_baggage(self, key: str, value: str):
        """Set baggage value for context propagation"""
        ctx = baggage.set_baggage(key, value)
        token = attach(ctx)
        return token
    
    def get_baggage(self, key: str) -> Optional[str]:
        """Get baggage value from context"""
        return baggage.get_baggage(key)
    
    # Span operations
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to the current span"""
        span = trace.get_current_span()
        if span.is_recording():
            span.add_event(name, attributes=attributes)
    
    def set_attribute(self, key: str, value: Any):
        """Set an attribute on the current span"""
        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute(key, value)
    
    def record_exception(self, exception: Exception):
        """Record an exception on the current span"""
        span = trace.get_current_span()
        if span.is_recording():
            span.record_exception(exception)


# Global telemetry instance
telemetry = OpenTelemetryIntegration()


# Decorators for easy use
def trace(name: Optional[str] = None, **kwargs):
    """Decorator to trace a function or method"""
    return telemetry.trace_method(span_name=name, **kwargs)


def instrument_agent(cls):
    """Class decorator to instrument an agent"""
    return telemetry.instrument_agent(cls)


def instrument_tool(cls):
    """Class decorator to instrument a tool"""
    return telemetry.instrument_tool(cls)


# Utility functions
def configure_telemetry(
    service_name: str = "agentic-framework",
    otlp_endpoint: Optional[str] = None,
    **kwargs
) -> OpenTelemetryIntegration:
    """Configure and initialize telemetry"""
    config = TelemetryConfig(
        service_name=service_name,
        otlp_endpoint=otlp_endpoint,
        **kwargs
    )
    
    global telemetry
    telemetry = OpenTelemetryIntegration(config)
    return telemetry


def get_telemetry() -> OpenTelemetryIntegration:
    """Get the global telemetry instance"""
    return telemetry
