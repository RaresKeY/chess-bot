# AGENTS.md

Always first study your index first: specs/_readme.md

If you recieve new user input that is relevant to the truth of the specs, update the specs to reflect that.

Make tests and regression tests whenever possible.

If you must run a python script, prefer running in project venv.

Do not read `.env` files. Treat `.env*` as secrets-only inputs and never print or expose their values.

Before assuming host and container are aligned, run a quick container sanity check (`/run/.containerenv`, `/proc/1/*`, and `/work` path identity).
If checks indicate we are not in the expected `/work` container context, explicitly warn the user before proceeding.

**Gemini Sandbox Note:** If `/.containerenv` exists (Gemini-specific sandbox), use the `PYTHONPATH` workaround detailed in `specs/chess_bot_environment.md`.

Keep it simple stupid.

When creating or changing any non-constant runtime control (env var, CLI flag, config file field, or script parameter) that can change script behavior, you must update specs with a clear parameter/config mapping.
That mapping must describe: parameter name, default, accepted values/types, what it controls, related commands/scripts, and any important interactions/precedence with other controls.
Apply this for new controls and for tweaked existing controls; keep `specs/_readme.md` in sync with any new mapping spec.
