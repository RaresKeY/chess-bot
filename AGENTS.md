# AGENTS.md

Always first study your index first: specs/_readme.md

If you recieve new user input that is relevant to the truth of the specs, update the specs to reflect that.

If you must run a python script, prefer running in project venv.

Before assuming host and container are aligned, run a quick container sanity check (`/run/.containerenv`, `/proc/1/*`, and `/work` path identity).
If checks indicate we are not in the expected `/work` container context, explicitly warn the user before proceeding.
