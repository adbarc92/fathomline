"""Adapters: the only code that imports a simulator or a policy library.

Each one implements the core's `Env` and `Policy` protocols over a third-party
stack. The core never imports from here; the CLI does, lazily, so a platform
without the stack still runs everything else.
"""
