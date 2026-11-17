from contextlib import contextmanager
from functools import wraps


@contextmanager
def mark_compiled_step_starts(torch_module):
    compiler = getattr(torch_module, "compiler", None)
    mark_step_begin = getattr(compiler, "cudagraph_mark_step_begin", None)
    original_compile = getattr(torch_module, "compile", None)

    if not callable(mark_step_begin) or not callable(original_compile):
        raise RuntimeError(
            "Compiled inference requires torch.compiler.cudagraph_mark_step_begin"
        )

    def compile_with_step_marker(function, *args, **kwargs):
        compiled_function = original_compile(function, *args, **kwargs)

        @wraps(function)
        def run_compiled(*function_args, **function_kwargs):
            mark_step_begin()
            return compiled_function(*function_args, **function_kwargs)

        return run_compiled

    torch_module.compile = compile_with_step_marker
    try:
        yield
    finally:
        torch_module.compile = original_compile
