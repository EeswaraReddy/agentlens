# Contributing to AgentLens

Thanks for your interest. AgentLens is small on purpose — the core has zero
dependencies — so contributions that keep it that way are especially welcome.

## Development setup

```bash
git clone https://github.com/EeswaraReddy/agentlens.git
cd agentlens
pip install -e ".[dev]"
pytest -q
```

All 25 tests should pass.

## Project layout

- `agentlens/` — library code (no dependencies in the core)
  - `adapters/` — integrations with agent SDKs (lazy-imported)
  - `providers/` — LLM provider clients
- `examples/` — runnable demos
- `tests/` — pytest suite
- `docs/` — blog post, social posts, launch notes

## Guidelines

- **Keep the core dependency-free.** New optional features go behind a
  `pyproject.toml` extra and lazy-import their dependency.
- **Match the existing style.** Type-hinted, small classes, docstrings on
  public functions.
- **Tests are required for new behavior.** Faked SDKs are fine (see
  `tests/test_strands_adapter.py` for an example).
- **One change per PR.** Easier to review and revert.

## Adding a new adapter

1. Create `agentlens/adapters/<framework>.py`.
2. Lazy-import the framework inside a factory function so AgentLens still
   imports without it installed.
3. Map the framework's lifecycle events to `tracer.start_span()` /
   `tracer.end_span()` calls.
4. Add a test that fakes the framework's base classes (no install required).

## Releasing

Maintainers only:

1. Bump version in `pyproject.toml` and `agentlens/__init__.py`.
2. Update `CHANGELOG.md`.
3. Tag the commit: `git tag v0.1.1 && git push --tags`.
4. GitHub Actions builds and publishes to PyPI via trusted publishing.
