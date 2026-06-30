import functools
import inspect
import sys

import sentry_sdk
from sentry_sdk.consts import OP, SPANSTATUS
from sentry_sdk.integrations import DidNotEnable, Integration, _check_minimum_version
from sentry_sdk.traces import SegmentSource
from sentry_sdk.tracing import TransactionSource
from sentry_sdk.tracing_utils import has_span_streaming_enabled
from sentry_sdk.utils import (
    event_from_exception,
    logger,
    package_version,
    qualname_from_function,
    reraise,
)

try:
    import ray  # type: ignore[import-not-found]
    from ray import remote
except ImportError:
    raise DidNotEnable("Ray not installed.")

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Optional

    from sentry_sdk.utils import ExcInfo


def _check_sentry_initialized() -> None:
    if sentry_sdk.get_client().is_active():
        return

    logger.debug(
        "[Tracing] Sentry not initialized in ray cluster worker, performance data will be discarded."
    )


def _insert_sentry_tracing_in_signature(func: "Callable[..., Any]") -> None:
    # Patching new_func signature to add the _sentry_tracing parameter to it
    # Ray later inspects the signature and finds the unexpected parameter otherwise
    signature = inspect.signature(func)
    params = list(signature.parameters.values())
    sentry_tracing_param = inspect.Parameter(
        "_sentry_tracing",
        kind=inspect.Parameter.KEYWORD_ONLY,
        default=None,
    )

    # Keyword only arguments are penultimate if function has variadic keyword arguments
    if params and params[-1].kind is inspect.Parameter.VAR_KEYWORD:
        params.insert(-1, sentry_tracing_param)
    else:
        params.append(sentry_tracing_param)

    func.__signature__ = signature.replace(parameters=params)  # type: ignore[attr-defined]


def _wrap_actor_init(method: "Callable[..., Any]") -> "Callable[..., Any]":
    # Wraps an actor __init__ for error capture only.
    # Ray calls __init__ directly during actor creation, not via _actor_method_call,
    # so there is no mechanism to propagate trace context. We only capture exceptions.
    # Async __init__ (used by Ray Serve actors) requires an async wrapper so that
    # Ray can await it correctly in the actor's event loop.
    if inspect.iscoroutinefunction(method):

        @functools.wraps(method)
        async def new_init_async(self: "Any", *args: "Any", **kwargs: "Any") -> None:
            _check_sentry_initialized()
            try:
                await method(self, *args, **kwargs)
            except Exception:
                exc_info = sys.exc_info()
                _capture_exception(exc_info)
                reraise(*exc_info)

        return new_init_async

    @functools.wraps(method)
    def new_init(self: "Any", *args: "Any", **kwargs: "Any") -> None:
        _check_sentry_initialized()
        try:
            method(self, *args, **kwargs)
        except Exception:
            exc_info = sys.exc_info()
            _capture_exception(exc_info)
            reraise(*exc_info)

    return new_init


def _wrap_actor_method(
    method: "Callable[..., Any]", class_name: str, method_name: str
) -> "Callable[..., Any]":
    # Wraps a public actor method to continue a distributed trace and create an
    # execution span/transaction on the worker side. Accepts _sentry_tracing as
    # a keyword-only arg injected by the caller-side patch.
    # Async methods (used by Ray Serve and async actors) require an async wrapper
    # so that Ray can await them correctly in the actor's event loop.
    # Sentry's start_span/start_transaction context managers are synchronous and
    # can be used with regular `with` inside async functions.
    span_name = f"{class_name}.{method_name}"

    if inspect.iscoroutinefunction(method):

        @functools.wraps(method)
        async def new_method_async(
            self: "Any",
            *args: "Any",
            _sentry_tracing: "Optional[dict[str, Any]]" = None,
            **kwargs: "Any",
        ) -> "Any":
            _check_sentry_initialized()

            span_streaming = has_span_streaming_enabled(sentry_sdk.get_client().options)
            if span_streaming:
                sentry_sdk.traces.continue_trace(_sentry_tracing or {})

                with sentry_sdk.traces.start_span(
                    name=span_name,
                    attributes={
                        "sentry.op": OP.QUEUE_TASK_RAY,
                        "sentry.origin": RayIntegration.origin,
                        "sentry.span.source": SegmentSource.TASK,
                    },
                    parent_span=None,
                ):
                    try:
                        return await method(self, *args, **kwargs)
                    except Exception:
                        exc_info = sys.exc_info()
                        _capture_exception(exc_info)
                        reraise(*exc_info)
            else:
                transaction = sentry_sdk.continue_trace(
                    _sentry_tracing or {},
                    op=OP.QUEUE_TASK_RAY,
                    name=span_name,
                    origin=RayIntegration.origin,
                    source=TransactionSource.TASK,
                )

                with sentry_sdk.start_transaction(transaction) as transaction:
                    try:
                        result = await method(self, *args, **kwargs)
                        transaction.set_status(SPANSTATUS.OK)
                    except Exception:
                        transaction.set_status(SPANSTATUS.INTERNAL_ERROR)
                        exc_info = sys.exc_info()
                        _capture_exception(exc_info)
                        reraise(*exc_info)

                    return result

        # Ray calls inspect.unwrap() before reading actor method signatures, which
        # follows __wrapped__ (set by functools.wraps) to the original method and
        # loses the _sentry_tracing parameter. Removing __wrapped__ makes Ray see
        # new_method_async's actual signature, which includes _sentry_tracing.
        del new_method_async.__wrapped__
        return new_method_async

    @functools.wraps(method)
    def new_method(
        self: "Any",
        *args: "Any",
        _sentry_tracing: "Optional[dict[str, Any]]" = None,
        **kwargs: "Any",
    ) -> "Any":
        _check_sentry_initialized()

        span_streaming = has_span_streaming_enabled(sentry_sdk.get_client().options)
        if span_streaming:
            sentry_sdk.traces.continue_trace(_sentry_tracing or {})

            with sentry_sdk.traces.start_span(
                name=span_name,
                attributes={
                    "sentry.op": OP.QUEUE_TASK_RAY,
                    "sentry.origin": RayIntegration.origin,
                    "sentry.span.source": SegmentSource.TASK,
                },
                parent_span=None,
            ):
                try:
                    return method(self, *args, **kwargs)
                except Exception:
                    exc_info = sys.exc_info()
                    _capture_exception(exc_info)
                    reraise(*exc_info)
        else:
            transaction = sentry_sdk.continue_trace(
                _sentry_tracing or {},
                op=OP.QUEUE_TASK_RAY,
                name=span_name,
                origin=RayIntegration.origin,
                source=TransactionSource.TASK,
            )

            with sentry_sdk.start_transaction(transaction) as transaction:
                try:
                    result = method(self, *args, **kwargs)
                    transaction.set_status(SPANSTATUS.OK)
                except Exception:
                    transaction.set_status(SPANSTATUS.INTERNAL_ERROR)
                    exc_info = sys.exc_info()
                    _capture_exception(exc_info)
                    reraise(*exc_info)

                return result

    # Ray calls inspect.unwrap() before reading actor method signatures, which follows
    # __wrapped__ (set by functools.wraps) to the original method and loses the
    # _sentry_tracing parameter. Removing __wrapped__ makes Ray see new_method's actual
    # signature, which already includes _sentry_tracing as a real keyword-only param.
    del new_method.__wrapped__
    return new_method


def _patch_actor_class(cls: type) -> None:
    # Wraps each public method and __init__ of a Ray actor class in-place so that
    # the wrapped versions are seen by Ray's signature inspection at decoration time.
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name == "__init__":
            setattr(cls, name, _wrap_actor_init(method))
        elif not name.startswith("_"):
            setattr(cls, name, _wrap_actor_method(method, cls.__name__, name))


def _patch_actor_handle(handle: "Any", class_name: str) -> None:
    # Patches _actor_method_call on a specific ActorHandle instance so that each
    # actor method invocation creates a submit span and propagates trace headers.
    old_actor_method_call = handle._actor_method_call

    def new_actor_method_call(
        method_name: str,
        args: "Optional[list[Any]]" = None,
        kwargs: "Optional[dict[str, Any]]" = None,
        **opts: "Any",
    ) -> "Any":
        span_name = f"{class_name}.{method_name}"
        kwargs = kwargs or {}

        span_streaming = has_span_streaming_enabled(sentry_sdk.get_client().options)
        if span_streaming:
            with sentry_sdk.traces.start_span(
                name=span_name,
                attributes={
                    "sentry.op": OP.QUEUE_SUBMIT_RAY,
                    "sentry.origin": RayIntegration.origin,
                },
            ):
                tracing = {
                    k: v
                    for k, v in sentry_sdk.get_current_scope().iter_trace_propagation_headers()
                }
                try:
                    return old_actor_method_call(
                        method_name,
                        args=args,
                        kwargs={**kwargs, "_sentry_tracing": tracing},
                        **opts,
                    )
                except Exception:
                    exc_info = sys.exc_info()
                    _capture_exception(exc_info)
                    reraise(*exc_info)
        else:
            with sentry_sdk.start_span(
                op=OP.QUEUE_SUBMIT_RAY,
                name=span_name,
                origin=RayIntegration.origin,
            ) as span:
                tracing = {
                    k: v
                    for k, v in sentry_sdk.get_current_scope().iter_trace_propagation_headers()
                }
                try:
                    result = old_actor_method_call(
                        method_name,
                        args=args,
                        kwargs={**kwargs, "_sentry_tracing": tracing},
                        **opts,
                    )
                    span.set_status(SPANSTATUS.OK)
                except Exception:
                    span.set_status(SPANSTATUS.INTERNAL_ERROR)
                    exc_info = sys.exc_info()
                    _capture_exception(exc_info)
                    reraise(*exc_info)

                return result

    handle._actor_method_call = new_actor_method_call


def _wrap_actor_class_remote(actor_class: "Any", class_name: str) -> None:
    # Wraps actor_class.remote so each returned ActorHandle is patched for
    # caller-side tracing via _patch_actor_handle.
    old_class_remote = actor_class.remote

    def new_class_remote(*args: "Any", **kwargs: "Any") -> "Any":
        handle = old_class_remote(*args, **kwargs)
        _patch_actor_handle(handle, class_name)
        return handle

    actor_class.remote = new_class_remote


def _patch_ray_remote() -> None:
    old_remote = remote

    @functools.wraps(old_remote)
    def new_remote(
        f: "Optional[Callable[..., Any]]" = None, *args: "Any", **kwargs: "Any"
    ) -> "Callable[..., Any]":
        if inspect.isclass(f):
            # Ray Actors (https://docs.ray.io/en/latest/ray-core/actors.html)
            _patch_actor_class(f)
            actor_class = old_remote(f, *args, **kwargs)
            _wrap_actor_class_remote(actor_class, f.__name__)
            return actor_class

        def wrapper(user_f: "Callable[..., Any]") -> "Any":
            if inspect.isclass(user_f):
                # Ray Actors (https://docs.ray.io/en/latest/ray-core/actors.html)
                _patch_actor_class(user_f)
                actor_class = old_remote(*args, **kwargs)(user_f)
                _wrap_actor_class_remote(actor_class, user_f.__name__)
                return actor_class

            @functools.wraps(user_f)
            def new_func(
                *f_args: "Any",
                _sentry_tracing: "Optional[dict[str, Any]]" = None,
                **f_kwargs: "Any",
            ) -> "Any":
                _check_sentry_initialized()

                span_streaming = has_span_streaming_enabled(
                    sentry_sdk.get_client().options
                )
                if span_streaming:
                    sentry_sdk.traces.continue_trace(_sentry_tracing or {})

                    function_name = qualname_from_function(user_f)
                    with sentry_sdk.traces.start_span(
                        name="unknown Ray task"
                        if function_name is None
                        else function_name,
                        attributes={
                            "sentry.op": OP.QUEUE_TASK_RAY,
                            "sentry.origin": RayIntegration.origin,
                            "sentry.span.source": SegmentSource.TASK,
                        },
                        parent_span=None,
                    ):
                        try:
                            result = user_f(*f_args, **f_kwargs)
                        except Exception:
                            exc_info = sys.exc_info()
                            _capture_exception(exc_info)
                            reraise(*exc_info)

                        return result
                else:
                    transaction = sentry_sdk.continue_trace(
                        _sentry_tracing or {},
                        op=OP.QUEUE_TASK_RAY,
                        name=qualname_from_function(user_f),
                        origin=RayIntegration.origin,
                        source=TransactionSource.TASK,
                    )

                    with sentry_sdk.start_transaction(transaction) as transaction:
                        try:
                            result = user_f(*f_args, **f_kwargs)
                            transaction.set_status(SPANSTATUS.OK)
                        except Exception:
                            transaction.set_status(SPANSTATUS.INTERNAL_ERROR)
                            exc_info = sys.exc_info()
                            _capture_exception(exc_info)
                            reraise(*exc_info)

                        return result

            _insert_sentry_tracing_in_signature(new_func)

            if f:
                rv = old_remote(new_func)
            else:
                rv = old_remote(*args, **kwargs)(new_func)
            old_remote_method = rv.remote

            def _remote_method_with_header_propagation(
                *args: "Any", **kwargs: "Any"
            ) -> "Any":
                """
                Ray Client
                """
                span_streaming = has_span_streaming_enabled(
                    sentry_sdk.get_client().options
                )
                if span_streaming:
                    function_name = qualname_from_function(user_f)
                    with sentry_sdk.traces.start_span(
                        name="unknown Ray task"
                        if function_name is None
                        else function_name,
                        attributes={
                            "sentry.op": OP.QUEUE_SUBMIT_RAY,
                            "sentry.origin": RayIntegration.origin,
                        },
                    ):
                        tracing = {
                            k: v
                            for k, v in sentry_sdk.get_current_scope().iter_trace_propagation_headers()
                        }
                        try:
                            result = old_remote_method(
                                *args, **kwargs, _sentry_tracing=tracing
                            )
                        except Exception:
                            exc_info = sys.exc_info()
                            _capture_exception(exc_info)
                            reraise(*exc_info)

                        return result
                else:
                    with sentry_sdk.start_span(
                        op=OP.QUEUE_SUBMIT_RAY,
                        name=qualname_from_function(user_f),
                        origin=RayIntegration.origin,
                    ) as span:
                        tracing = {
                            k: v
                            for k, v in sentry_sdk.get_current_scope().iter_trace_propagation_headers()
                        }
                        try:
                            result = old_remote_method(
                                *args, **kwargs, _sentry_tracing=tracing
                            )
                            span.set_status(SPANSTATUS.OK)
                        except Exception:
                            span.set_status(SPANSTATUS.INTERNAL_ERROR)
                            exc_info = sys.exc_info()
                            _capture_exception(exc_info)
                            reraise(*exc_info)

                        return result

            rv.remote = _remote_method_with_header_propagation

            return rv

        if f is not None:
            return wrapper(f)
        else:
            return wrapper

    ray.remote = new_remote


def _capture_exception(exc_info: "ExcInfo", **kwargs: "Any") -> None:
    client = sentry_sdk.get_client()

    event, hint = event_from_exception(
        exc_info,
        client_options=client.options,
        mechanism={
            "handled": False,
            "type": RayIntegration.identifier,
        },
    )
    sentry_sdk.capture_event(event, hint=hint)


class RayIntegration(Integration):
    identifier = "ray"
    origin = f"auto.queue.{identifier}"

    @staticmethod
    def setup_once() -> None:
        version = package_version("ray")
        _check_minimum_version(RayIntegration, version)

        _patch_ray_remote()
