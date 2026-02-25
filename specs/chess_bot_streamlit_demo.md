# Chess Bot Streamlit Demo Component

## Responsibility
Provide a standalone Streamlit app for playing chess against a pretrained next-move model artifact, designed to be moved/copied as an independent repo root.

## Code Ownership
- App entrypoint: `streamlit_chess_ai_demo/app.py`
- Demo dependencies: `streamlit_chess_ai_demo/requirements.txt`
- Deployment config: `streamlit_chess_ai_demo/.streamlit/config.toml`
- Model drop-in directory: `streamlit_chess_ai_demo/models/`
- Usage docs: `streamlit_chess_ai_demo/README.md`

## Standalone Packaging Contract
- Folder is self-contained and does not import from parent repo modules (`src/chessbot/*`)
- Includes minimal inference/runtime/model code required to load project-trained `.pt` artifacts
- Uses CPU inference (`torch.load(..., map_location="cpu")`) for portability

## UI Behavior (current)
- Displays board using `python-chess` SVG rendering inside Streamlit
- User selects a legal move from a dropdown and submits it
- App applies model reply on server side and shows move history + top-k predictions
- `New Game` and `Undo Pair` controls manage session state

## Model Swapping (current)
- Auto-discovers local `*.pt` files under `streamlit_chess_ai_demo/models/`
- Sidebar file uploader can temporarily override local model selection with an uploaded `.pt`
- Compatible artifacts must match training output structure (`state_dict`, `vocab`, `config`)

## Known Limitations (current)
- UI is move-select based (not click-to-move board interaction)
- Black-side play flow is not optimized; current UX is primarily for user-as-White
- Large Torch installs may increase Streamlit Cloud cold-start time
