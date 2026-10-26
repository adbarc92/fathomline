# Contributing

Pull requests are welcome. For anything larger than a small fix, open an issue first so the change
can be checked against the decision register in [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Setup

Fathomline uses [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync --group dev
```

The real fixture (LeRobot on LIBERO) is Linux-only and runs in the container image
(`docker build -t fathomline .`). Everything else, including the whole unit suite, runs against the
fake environment on any platform.

## The gates

Every change must pass the same checks CI runs:

```bash
uv run pytest --cov=fathomline --cov-report=term-missing   # tests, 100% line and branch coverage
uv run ruff check .                                         # lint
uv run ruff format --check .                                # formatting
uv run mypy                                                 # strict type checking
```

- Write the test first, and see it fail before it passes.
- A new README claim needs a proving command in the README's table.
- Every `# pragma: no cover` carries its reason on the same line.

Agents and humans follow the same gates: an AI-authored PR is held to the same tests, coverage, lint
and review as any other.
