# Chess Bot Regression Testing

## Responsibility
Define how to add regression tests after a bug fix so the same failure mode is covered permanently and future changes do not reintroduce it.

## Scope
- Applies to all `chess_bot` components (`scripts/*`, `src/chessbot/*`)
- Covers unit/integration-style regressions in local test suites
- Focuses on test additions tied to a specific observed bug or bad flow

## When To Add A Regression Test
- A real runtime bug was observed (traceback, wrong output, bad API call, stuck flow)
- A user reported a behavior regression
- A bug fix changes branching logic or edge-case handling
- A live/manual smoke test exposed a failure that can be simulated locally

Do not skip the regression test when:
- The bug is deterministic and reproducible with local inputs/mocks
- The bug affects external API interactions but can be mocked

## Regression Test Pattern (required)
1. Reproduce the failing scenario in a test using the smallest possible setup.
2. Assert the pre-fix bad behavior would fail this test (implicitly via the new assertion).
3. Apply the code fix.
4. Run the targeted test file first.
5. Keep the test focused on the bug contract, not unrelated behavior.

## Placement Rules
- Prefer existing component test file when one already covers the code path.
  - Example: Lichess bot bugs go in `tests/test_lichess_bot.py`
- Create a new test file only when no existing file matches the component.
- Keep test names bug-specific and behavior-oriented:
  - Good: `test_runner_ignores_outbound_challenge_events`
  - Bad: `test_bugfix_1`

## Test Design Guidelines
- Use mocks/fakes for network and external services.
- Avoid real network calls in regression tests.
- Prefer deterministic fixtures over dynamic data.
- Assert the key contract that broke:
  - no exception raised
  - correct branch chosen
  - no invalid API call issued
  - expected event/log emitted
  - expected file/schema field written

## Lichess Bot Regression Example (current)
- Bug observed: outbound challenge events (`challenge.direction == "out"`) were treated like incoming challenges, causing an invalid accept call and `HTTP 404`.
- Fix location: `src/chessbot/lichess_bot.py`
- Regression test added: `tests/test_lichess_bot.py`
  - `test_runner_ignores_outbound_challenge_events`
  - Asserts no accept/decline call occurs and an informational log event is emitted.

## Minimum Checklist For Every Regression Fix
- Add or update a test that fails without the fix.
- Run the targeted test file (project venv preferred).
- If the bug was found in a live/manual flow, perform a short smoke test of that flow after the fix.
- Update the relevant component spec if the fix changes behavior/contract.

## Command Guidance
- Prefer project venv:
  - `.venv/bin/python -m unittest -q tests.test_...`
- If a test file uses `pytest`, use:
  - `.venv/bin/python -m pytest -q tests/test_...py`
- For full-suite convenience, use:
  - `bash scripts/run_all_tests.sh`

## Anti-Patterns
- Fixing the bug without a test when a local test is feasible
- Adding a broad snapshot/assertion that does not verify the bug contract
- Using real external API calls in unit regressions
- Coupling the regression test to unrelated implementation details
