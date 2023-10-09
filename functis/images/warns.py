import inspect
from warnings import warn

warn_cache = set()

def warn_once(warning):
    """
    Warns only once per line of code instead of spamming the user with the same warning.
    """
    call_stack = inspect.stack()
    ref = f"{call_stack[1].function}:{call_stack[1].lineno}:{call_stack[1].code_context[0]}"
    if ref not in warn_cache:
        warn_cache.add(ref)
        warn(warning, UserWarning)
