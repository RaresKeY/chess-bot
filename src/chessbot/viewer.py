import html
import json
from typing import Dict, List, Optional

import chess
import chess.pgn


def load_game_positions_from_pgn(pgn_path: str, game_index: int = 0) -> Dict:
    with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
        game: Optional[chess.pgn.Game] = None
        for idx in range(game_index + 1):
            game = chess.pgn.read_game(f)
            if game is None:
                raise ValueError(f"No game at index {game_index} in {pgn_path}")

    assert game is not None
    board = game.board()
    positions: List[Dict] = [
        {
            "fen": board.fen(),
            "ply": 0,
            "san": "",
            "uci": "",
            "turn": "white",
            "fullmove": 1,
        }
    ]

    for ply, move in enumerate(game.mainline_moves(), start=1):
        san = board.san(move)
        uci = move.uci()
        board.push(move)
        positions.append(
            {
                "fen": board.fen(),
                "ply": ply,
                "san": san,
                "uci": uci,
                "turn": "white" if board.turn == chess.WHITE else "black",
                "fullmove": board.fullmove_number,
            }
        )

    headers = {k: v for k, v in game.headers.items()}
    moves = []
    for i in range(1, len(positions)):
        moves.append(
            {
                "ply": positions[i]["ply"],
                "san": positions[i]["san"],
                "uci": positions[i]["uci"],
            }
        )

    return {
        "headers": headers,
        "positions": positions,
        "moves": moves,
    }


def _template_html(payload: Dict, piece_base: str) -> str:
    data_json = json.dumps(payload, ensure_ascii=True)
    piece_base_js = json.dumps(piece_base.rstrip("/"))
    title = html.escape(payload.get("headers", {}).get("Event", "Chess Game Viewer"))
    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>{title}</title>
<style>
:root {{
  --bg: #f5f1e8;
  --ink: #1e1a16;
  --panel: #fffaf0;
  --accent: #22543d;
  --light: #f0d9b5;
  --dark: #b58863;
  --border: #d8ccb8;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: radial-gradient(circle at 20% 10%, #fff7dc, #efe4ce 55%, #e4d4be);
  color: var(--ink);
  font-family: Georgia, "Times New Roman", serif;
}}
.wrap {{
  max-width: 1100px;
  margin: 24px auto;
  padding: 16px;
  display: grid;
  grid-template-columns: minmax(280px, 560px) minmax(260px, 1fr);
  gap: 18px;
}}
@media (max-width: 860px) {{
  .wrap {{ grid-template-columns: 1fr; }}
}}
.panel {{
  background: color-mix(in srgb, var(--panel) 95%, white);
  border: 1px solid var(--border);
  border-radius: 16px;
  box-shadow: 0 8px 30px rgba(40, 30, 15, 0.08);
}}
.board-shell {{ padding: 14px; }}
.board {{
  width: min(100%, 540px);
  aspect-ratio: 1;
  display: grid;
  grid-template-columns: repeat(8, 1fr);
  grid-template-rows: repeat(8, 1fr);
  border: 2px solid #6d5234;
  margin: 0 auto;
  overflow: hidden;
  border-radius: 10px;
}}
.square {{ position: relative; aspect-ratio: 1 / 1; }}
.square.light {{ background: var(--light); }}
.square.dark {{ background: var(--dark); }}
.square .piece {{ width: 100%; height: 100%; display: block; padding: 4%; }}
.square .coord {{
  position: absolute; font-size: 11px; opacity: 0.7; font-weight: 700;
}}
.square .coord.file {{ right: 4px; bottom: 2px; }}
.square .coord.rank {{ left: 4px; top: 2px; }}
.controls {{
  margin-top: 12px;
  display: grid;
  grid-template-columns: repeat(4, auto) 1fr;
  gap: 8px;
  align-items: center;
}}
button {{
  border: 1px solid #8b6a44;
  background: linear-gradient(#fff8ee, #efe0c8);
  color: #2d2115;
  border-radius: 10px;
  padding: 8px 12px;
  font-size: 14px;
  cursor: pointer;
}}
button:hover {{ filter: brightness(0.98); }}
button:active {{ transform: translateY(1px); }}
.status {{ font-size: 14px; text-align: right; color: #4b3a29; }}
.side {{ padding: 14px; display: grid; gap: 12px; align-content: start; }}
.card {{
  background: #fffdf8;
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px;
}}
.hdr {{ margin: 0 0 8px; font-size: 16px; color: var(--accent); }}
.meta {{ display: grid; gap: 4px; font-size: 14px; }}
.meta-row {{ display: grid; grid-template-columns: 84px 1fr; gap: 8px; }}
.meta-row b {{ color: #5a4630; }}
.move-line {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }}
.moves {{
  max-height: 520px;
  overflow: auto;
  display: grid;
  gap: 4px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 13px;
}}
.move-row {{
  display: grid;
  grid-template-columns: 42px 1fr 1fr;
  gap: 8px;
  padding: 4px 6px;
  border-radius: 8px;
}}
.move-row.active {{ background: #efe7d7; outline: 1px solid #d8ccb8; }}
.move-pill {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.hint {{ font-size: 12px; opacity: 0.75; }}
</style>
</head>
<body>
<div class=\"wrap\">
  <section class=\"panel board-shell\">
    <div id=\"board\" class=\"board\"></div>
    <div class=\"controls\">
      <button id=\"firstBtn\" title=\"Home\">|&lt;</button>
      <button id=\"prevBtn\" title=\"Left Arrow\">&larr;</button>
      <button id=\"nextBtn\" title=\"Right Arrow\">&rarr;</button>
      <button id=\"lastBtn\" title=\"End\">&gt;|</button>
      <div id=\"status\" class=\"status\"></div>
    </div>
  </section>
  <aside class=\"panel side\">
    <section class=\"card\">
      <h2 class=\"hdr\">Game</h2>
      <div class=\"meta\" id=\"meta\"></div>
      <div class=\"hint\">Keyboard: Left/Right arrows to navigate</div>
    </section>
    <section class=\"card\">
      <h2 class=\"hdr\">Current Move</h2>
      <div id=\"currentMove\" class=\"move-line\">Start position</div>
    </section>
    <section class=\"card\">
      <h2 class=\"hdr\">Moves</h2>
      <div id=\"moves\" class=\"moves\"></div>
    </section>
  </aside>
</div>
<script>
const DATA = {data_json};
const PIECE_BASE = {piece_base_js};
const boardEl = document.getElementById('board');
const statusEl = document.getElementById('status');
const metaEl = document.getElementById('meta');
const currentMoveEl = document.getElementById('currentMove');
const movesEl = document.getElementById('moves');
const files = ['a','b','c','d','e','f','g','h'];
let idx = 0;

function pieceToAsset(ch) {{
  const isWhite = ch === ch.toUpperCase();
  const map = {{ 'k':'K','q':'Q','r':'R','b':'B','n':'N','p':'P' }};
  const key = map[ch.toLowerCase()];
  if (!key) return '';
  return `${{PIECE_BASE}}/${{isWhite ? 'w' : 'b'}}${{key}}.svg`;
}}

function fenBoard(fen) {{
  return fen.split(' ')[0].split('/');
}}

function expandRank(rank) {{
  const out = [];
  for (const c of rank) {{
    if (/\\d/.test(c)) {{
      for (let i = 0; i < Number(c); i++) out.push('');
    }} else {{
      out.push(c);
    }}
  }}
  return out;
}}

function buildMeta() {{
  const h = DATA.headers || {{}};
  const rows = [
    ['White', h.White || '?'],
    ['Black', h.Black || '?'],
    ['Event', h.Event || '?'],
    ['Date', h.Date || '?'],
    ['Result', h.Result || '?']
  ];
  metaEl.innerHTML = rows.map(([k,v]) => `<div class=\"meta-row\"><b>${{k}}</b><span>${{String(v)}}</span></div>`).join('');
}}

function buildMovesList() {{
  const moves = DATA.moves || [];
  const rows = [];
  for (let i = 0; i < moves.length; i += 2) {{
    const white = moves[i];
    const black = moves[i + 1];
    const moveNo = Math.floor(i / 2) + 1;
    rows.push(`
      <div class=\"move-row\" data-start-ply=\"${{white ? white.ply : 0}}\" data-end-ply=\"${{black ? black.ply : (white ? white.ply : 0)}}\">
        <div>${{moveNo}}.</div>
        <div class=\"move-pill\" data-ply=\"${{white ? white.ply : 0}}\">${{white ? white.san : ''}}</div>
        <div class=\"move-pill\" data-ply=\"${{black ? black.ply : 0}}\">${{black ? black.san : ''}}</div>
      </div>
    `);
  }}
  movesEl.innerHTML = rows.join('');
  movesEl.querySelectorAll('[data-ply]').forEach((el) => {{
    el.addEventListener('click', () => {{
      const ply = Number(el.getAttribute('data-ply') || '0');
      if (ply >= 0) setIndex(ply);
    }});
  }});
}}

function renderBoard() {{
  const pos = DATA.positions[idx];
  const ranks = fenBoard(pos.fen);
  let html = '';
  for (let r = 0; r < 8; r++) {{
    const rankNum = 8 - r;
    const cells = expandRank(ranks[r]);
    for (let c = 0; c < 8; c++) {{
      const sqColor = (r + c) % 2 === 0 ? 'light' : 'dark';
      const piece = cells[c];
      const showRank = c === 0;
      const showFile = r === 7;
      html += `<div class=\"square ${{sqColor}}\">`;
      if (piece) {{
        html += `<img class=\"piece\" alt=\"${{piece}}\" src=\"${{pieceToAsset(piece)}}\"/>`;
      }}
      if (showRank) html += `<span class=\"coord rank\">${{rankNum}}</span>`;
      if (showFile) html += `<span class=\"coord file\">${{files[c]}}</span>`;
      html += `</div>`;
    }}
  }}
  boardEl.innerHTML = html;
}}

function renderStatus() {{
  const pos = DATA.positions[idx];
  statusEl.textContent = `Ply ${{pos.ply}} / ${{DATA.positions.length - 1}}`;
  if (idx === 0) {{
    currentMoveEl.textContent = 'Start position';
  }} else {{
    currentMoveEl.textContent = `${{pos.ply}}. ${{pos.san}} (${{pos.uci}})`;
  }}
  movesEl.querySelectorAll('.move-row').forEach((row) => row.classList.remove('active'));
  const activeRow = Array.from(movesEl.querySelectorAll('.move-row')).find((row) => {{
    const start = Number(row.getAttribute('data-start-ply') || '0');
    const end = Number(row.getAttribute('data-end-ply') || '0');
    return idx >= start && idx <= end && idx !== 0;
  }});
  if (activeRow) activeRow.classList.add('active');
}}

function setIndex(next) {{
  idx = Math.max(0, Math.min(DATA.positions.length - 1, next));
  renderBoard();
  renderStatus();
}}

buildMeta();
buildMovesList();
setIndex(0);

document.getElementById('firstBtn').addEventListener('click', () => setIndex(0));
document.getElementById('prevBtn').addEventListener('click', () => setIndex(idx - 1));
document.getElementById('nextBtn').addEventListener('click', () => setIndex(idx + 1));
document.getElementById('lastBtn').addEventListener('click', () => setIndex(DATA.positions.length - 1));
window.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowLeft') setIndex(idx - 1);
  if (e.key === 'ArrowRight') setIndex(idx + 1);
  if (e.key === 'Home') setIndex(0);
  if (e.key === 'End') setIndex(DATA.positions.length - 1);
}});
</script>
</body>
</html>
"""


def render_game_viewer_html(pgn_path: str, out_html: str, game_index: int = 0, piece_base: str = "../assets/pieces/cburnett") -> Dict:
    payload = load_game_positions_from_pgn(pgn_path, game_index=game_index)
    html_text = _template_html(payload=payload, piece_base=piece_base)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_text)
    return {
        "out_html": out_html,
        "game_index": game_index,
        "positions": len(payload["positions"]),
        "headers": payload["headers"],
    }
