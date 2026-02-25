# Commit Workflow

When asked to commit, follow this sequence exactly.

## 1) Validate changes

- Run checks relevant to changed files (`test`, `lint`, `typecheck`).
- If any check fails, do not commit. Report failures and stop.

## 2) Review staged scope only

- Show staged files.
- Summarize staged diff only.
- Exclude unstaged changes from commit-message reasoning.

## 3) Message quality

- Use Conventional Commits: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`.
- Keep subject line imperative and <= 72 chars.
- Body must include: why, key behavior changes, and risk/migration notes (if any).

## 4) Safe commit behavior

- Never amend unless explicitly requested.
- Never commit unrelated files.
- Never use destructive git commands (`reset --hard`, checkout file reverts) unless explicitly requested.

## 5) Skill routing

- Draft message only: `staged-commit-message`.
- Commit staged changes: `elevated-staged-commit`.
- Commit + tag/release flow: `git-release-orchestrator`.

## 6) Pre-commit report

Before finalizing, report:

- Branch name.
- Staged files.
- Commit title.
- Validation commands with pass/fail status.
