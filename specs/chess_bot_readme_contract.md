# Chess Bot README Contract

## Purpose
Keep `README.md` navigable by separable components instead of a monolithic workflow description.

## Requirements
Each pipeline section in `README.md` must include:
- CLI entrypoint in `scripts/`
- Core implementation module(s) in `src/chessbot/`
- Primary produced artifacts (if any)

## Sync Rule
When logic moves between `scripts/` and `src/chessbot/`, update both:
- `README.md`
- the corresponding component spec in `specs/`

## Import Layout Note
- Keep `src/`, `src/chessbot/`, and `scripts/` package-initialized with `__init__.py` so tooling/tests can import modules reliably.
