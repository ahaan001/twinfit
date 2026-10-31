"""A stage module whose import fails for a reason other than being absent (used by a test)."""

import nonexistent_dependency_xyz  # noqa: F401


def run(ctx, options):
    raise NotImplementedError
