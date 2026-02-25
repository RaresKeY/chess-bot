import html
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import chess
import torch

from src.chessbot.inference import best_legal_from_topk
from src.chessbot.model import NextMoveLSTM, encode_tokens, winner_to_id


@dataclass
class PlayConfig:
    winner_side: str = "B"
    topk: int = 10
    user_color: str = "white"


class LoadedMoveModel:
    def __init__(self, artifact: Dict):
        self.artifact = artifact
        self.vocab = artifact["vocab"]
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}
        cfg = artifact["config"]
        self.model = NextMoveLSTM(vocab_size=len(self.vocab), **cfg)
        self.model.load_state_dict(artifact["state_dict"])
        self.model.eval()

    @classmethod
    def from_path(cls, model_path: str) -> "LoadedMoveModel":
        artifact = torch.load(model_path, map_location="cpu")
        return cls(artifact)

    def infer(self, context: List[str], winner_side: str, topk: int) -> Dict:
        context_ids = encode_tokens(context, self.vocab)
        if not context_ids:
            # If no context, produce a tensor with one unk token to avoid zero-length packed sequence.
            context_ids = [self.vocab.get("<UNK>", 1)]
        tokens = torch.tensor([context_ids], dtype=torch.long)
        lengths = torch.tensor([len(context_ids)], dtype=torch.long)
        winners = torch.tensor([winner_to_id(winner_side)], dtype=torch.long)

        with torch.no_grad():
            logits = self.model(tokens, lengths, winners)
            k = max(1, min(int(topk), logits.shape[-1]))
            pred_ids = logits.topk(k, dim=1).indices[0].tolist()
        topk_tokens = [self.inv_vocab.get(i, "") for i in pred_ids]
        legal = best_legal_from_topk(topk_tokens, context)
        return {"topk": topk_tokens, "best_legal": legal}


def board_from_context(context: List[str]) -> chess.Board:
    board = chess.Board()
    for i, uci in enumerate(context, start=1):
        try:
            mv = chess.Move.from_uci(uci)
        except Exception as exc:
            raise ValueError(f"Invalid UCI at ply {i}: {uci}") from exc
        if mv not in board.legal_moves:
            raise ValueError(f"Illegal context move at ply {i}: {uci}")
        board.push(mv)
    return board


def _move_list_with_san(context: List[str]) -> List[Dict]:
    board = chess.Board()
    out: List[Dict] = []
    for ply, uci in enumerate(context, start=1):
        mv = chess.Move.from_uci(uci)
        san = board.san(mv)
        board.push(mv)
        out.append({"ply": ply, "uci": uci, "san": san})
    return out


def serialize_state(context: List[str]) -> Dict:
    board = board_from_context(context)
    return {
        "context": context,
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "game_over": board.is_game_over(claim_draw=True),
        "result": board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*",
        "is_check": board.is_check(),
        "moves": _move_list_with_san(context),
    }


def _normalize_user_move(board: chess.Board, uci: str) -> chess.Move:
    try:
        mv = chess.Move.from_uci(uci)
    except Exception as exc:
        raise ValueError(f"Invalid move format: {uci}") from exc
    if mv in board.legal_moves:
        return mv

    # Auto-queen promotion fallback for 4-char pawn promotions.
    if len(uci) == 4:
        try_q = chess.Move.from_uci(uci + "q")
        if try_q in board.legal_moves:
            return try_q
    raise ValueError(f"Illegal move: {uci}")


def apply_user_and_model_move(
    model_runtime: LoadedMoveModel,
    context: List[str],
    user_move_uci: str,
    cfg: PlayConfig,
) -> Dict:
    board = board_from_context(context)
    expected_user_turn = chess.WHITE if cfg.user_color == "white" else chess.BLACK
    if board.turn != expected_user_turn:
        raise ValueError(f"It is not {cfg.user_color}'s turn")

    user_move = _normalize_user_move(board, user_move_uci)
    user_uci = user_move.uci()
    board.push(user_move)
    next_context = context + [user_uci]
    user_san = _move_list_with_san(next_context)[-1]["san"]

    model_reply = None
    if not board.is_game_over(claim_draw=True):
        infer = model_runtime.infer(next_context, winner_side=cfg.winner_side, topk=cfg.topk)
        reply_uci = infer.get("best_legal", "")
        if reply_uci:
            reply_move = chess.Move.from_uci(reply_uci)
            if reply_move in board.legal_moves:
                reply_san = board.san(reply_move)
                board.push(reply_move)
                next_context.append(reply_uci)
                model_reply = {
                    "uci": reply_uci,
                    "san": reply_san,
                    "topk": infer.get("topk", []),
                }
            else:
                model_reply = {"uci": "", "san": "", "topk": infer.get("topk", []), "error": "predicted move not legal"}
        else:
            model_reply = {"uci": "", "san": "", "topk": infer.get("topk", []), "error": "no legal model move"}

        if (not model_reply or not model_reply.get("uci")) and not board.is_game_over(claim_draw=True):
            fallback_move = next(iter(board.legal_moves), None)
            if fallback_move is not None:
                fallback_san = board.san(fallback_move)
                board.push(fallback_move)
                next_context.append(fallback_move.uci())
                model_reply = {
                    "uci": fallback_move.uci(),
                    "san": fallback_san,
                    "topk": (model_reply or {}).get("topk", []),
                    "fallback": True,
                    "error": (model_reply or {}).get("error", "") or "model fallback used",
                }

    state = serialize_state(next_context)
    state["last_user_move"] = {"uci": user_uci, "san": user_san}
    state["last_model_move"] = model_reply
    return state


def render_play_page_html(title: str, piece_base: str, default_winner_side: str, default_topk: int) -> str:
    title = html.escape(title)
    piece_base_js = json.dumps(piece_base.rstrip("/"))
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
  --warn: #a83b2f;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; color: var(--ink); font-family: Georgia, \"Times New Roman\", serif;
  background: radial-gradient(circle at 20% 10%, #fff7dc, #efe4ce 55%, #e4d4be);
}}
.wrap {{ max-width: 1160px; margin: 18px auto; padding: 14px; display:grid; grid-template-columns:minmax(280px,560px) minmax(280px,1fr); gap:16px; }}
@media (max-width: 900px) {{ .wrap {{ grid-template-columns: 1fr; }} }}
.panel {{ background:#fffaf0f2; border:1px solid var(--border); border-radius:16px; box-shadow:0 8px 30px rgba(40,30,15,.08); }}
.board-shell {{ padding: 14px; }}
.board {{ width:min(100%,540px); aspect-ratio:1; display:grid; grid-template-columns:repeat(8,1fr); grid-template-rows:repeat(8,1fr); border:2px solid #6d5234; border-radius:10px; overflow:hidden; margin:0 auto; }}
.square {{ position:relative; aspect-ratio:1/1; cursor:pointer; }}
.square.light {{ background:var(--light); }} .square.dark {{ background:var(--dark); }}
.square.sel {{ outline: 3px solid #2f855a; outline-offset: -3px; }}
.square.hint::after {{ content:''; position:absolute; inset:36%; border-radius:50%; background:rgba(34,84,61,.45); }}
.square .piece {{ width:100%; height:100%; display:block; padding:4%; pointer-events:none; }}
.square .coord {{ position:absolute; font-size:11px; opacity:.7; font-weight:700; }}
.square .coord.file {{ right:4px; bottom:2px; }} .square .coord.rank {{ left:4px; top:2px; }}
.controls {{ margin-top:12px; display:grid; grid-template-columns: repeat(4, auto) 1fr auto auto; gap:8px; align-items:center; }}
button, select {{ border:1px solid #8b6a44; background:linear-gradient(#fff8ee,#efe0c8); color:#2d2115; border-radius:10px; padding:8px 10px; font-size:14px; }}
button {{ cursor:pointer; }} button:disabled {{ opacity:.45; cursor:not-allowed; }}
.status {{ font-size:14px; text-align:right; color:#4b3a29; }}
.side {{ padding:14px; display:grid; gap:12px; align-content:start; }}
.card {{ background:#fffdf8; border:1px solid var(--border); border-radius:12px; padding:12px; }}
.hdr {{ margin:0 0 8px; font-size:16px; color:var(--accent); }}
.row {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
.small {{ font-size:12px; opacity:.8; }}
.error {{ color:var(--warn); font-weight:700; min-height:1.2em; }}
.moves {{ max-height:480px; overflow:auto; display:grid; gap:4px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:13px; }}
.move-row {{ display:grid; grid-template-columns:42px 1fr 1fr; gap:8px; padding:4px 6px; border-radius:8px; }}
.move-row.active {{ background:#efe7d7; outline:1px solid #d8ccb8; }}
.meta-grid {{ display:grid; grid-template-columns: 90px 1fr; gap:6px 8px; font-size:14px; }}
.log-toolbar {{ display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:8px; }}
.log-panel {{ max-height:180px; overflow:auto; display:grid; gap:6px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; }}
.log-panel.hidden {{ display:none; }}
.log-entry {{ padding:6px 8px; border-radius:8px; border:1px solid var(--border); background:#fffaf0; }}
.log-entry.error {{ color:#7f1d1d; background:#fff1ee; border-color:#e6b8b1; font-weight:600; }}
.log-entry.info {{ color:#3f3124; }}
</style>
</head>
<body>
<div class=\"wrap\">
  <section class=\"panel board-shell\">
    <div id=\"board\" class=\"board\"></div>
    <div class=\"controls\">
      <button id=\"firstBtn\">|&lt;</button>
      <button id=\"prevBtn\">&larr;</button>
      <button id=\"nextBtn\">&rarr;</button>
      <button id=\"lastBtn\">&gt;|</button>
      <div id=\"status\" class=\"status\"></div>
      <button id=\"undoBtn\">Undo Pair</button>
      <button id=\"newBtn\">New Game</button>
    </div>
  </section>
  <aside class=\"panel side\">
    <section class=\"card\">
      <h2 class=\"hdr\">Play vs Model</h2>
      <div class=\"row\">
        <label for=\"winnerSide\">winner_side</label>
        <select id=\"winnerSide\">
          <option value=\"W\">W</option>
          <option value=\"B\">B</option>
          <option value=\"D\">D</option>
          <option value=\"?\">?</option>
        </select>
      </div>
      <div class=\"small\">You play White. Click source square, then destination square.</div>
      <div id=\"error\" class=\"error\"></div>
    </section>
    <section class=\"card\">
      <h2 class=\"hdr\">State</h2>
      <div class=\"meta-grid\">
        <b>Turn</b><span id=\"turnTxt\">white</span>
        <b>Result</b><span id=\"resultTxt\">*</span>
        <b>Selected</b><span id=\"selectedTxt\">none</span>
        <b>Top-k</b><span>{default_topk}</span>
      </div>
      <div class=\"small\">Keyboard viewer nav: Left/Right/Home/End (history only)</div>
    </section>
    <section class=\"card\">
      <h2 class=\"hdr\">Moves</h2>
      <div id=\"moves\" class=\"moves\"></div>
    </section>
    <section class=\"card\">
      <div class=\"log-toolbar\">
        <h2 class=\"hdr\" style=\"margin:0;\">Log</h2>
        <button id=\"toggleLogBtn\" type=\"button\">Hide Log</button>
      </div>
      <div id=\"logPanel\" class=\"log-panel\"></div>
    </section>
  </aside>
</div>
<script>
const PIECE_BASE = {piece_base_js};
const DEFAULTS = {{ winnerSide: {json.dumps(default_winner_side)}, topk: {default_topk} }};
const files = ['a','b','c','d','e','f','g','h'];
const boardEl = document.getElementById('board');
const statusEl = document.getElementById('status');
const movesEl = document.getElementById('moves');
const errEl = document.getElementById('error');
const logPanelEl = document.getElementById('logPanel');
const toggleLogBtn = document.getElementById('toggleLogBtn');
const turnTxt = document.getElementById('turnTxt');
const resultTxt = document.getElementById('resultTxt');
const selectedTxt = document.getElementById('selectedTxt');
const winnerSideSel = document.getElementById('winnerSide');
winnerSideSel.value = DEFAULTS.winnerSide;

const state = {{
  context: [],
  selected: null,
  fen: 'start',
  moves: [],
  snapshots: [],
  snapshotIndex: 0,
  turn: 'white',
  result: '*',
  game_over: false,
  legalTargets: [],
  logs: [],
  logVisible: true
}};

function pieceToAsset(ch) {{
  const isWhite = ch === ch.toUpperCase();
  const map = {{ 'k':'K','q':'Q','r':'R','b':'B','n':'N','p':'P' }};
  const key = map[ch.toLowerCase()];
  if (!key) return '';
  return `${{PIECE_BASE}}/${{isWhite ? 'w' : 'b'}}${{key}}.svg`;
}}
function fenBoard(fen) {{ return fen.split(' ')[0].split('/'); }}
function expandRank(rank) {{
  const out = [];
  for (const c of rank) {{
    if (/\\d/.test(c)) {{ for (let i = 0; i < Number(c); i++) out.push(''); }}
    else out.push(c);
  }}
  return out;
}}
function squareName(r, c) {{ return files[c] + String(8 - r); }}
function sameSquare(a, b) {{ return a && b && a === b; }}

async function api(path, body) {{
  const res = await fetch(path, {{ method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(body || {{}}) }});
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${{res.status}}`);
  return data;
}}

function setError(msg) {{ errEl.textContent = msg || ''; }}

function pushLog(level, message) {{
  if (!message) return;
  state.logs.push({{
    ts: new Date().toLocaleTimeString(),
    level: String(level || 'INFO').toUpperCase(),
    message: String(message)
  }});
  if (state.logs.length > 200) state.logs = state.logs.slice(-200);
}}

function applyServerState(data) {{
  const prevContextLen = state.context.length;
  state.context = data.context || [];
  state.fen = data.fen;
  state.turn = data.turn;
  state.result = data.result;
  state.game_over = !!data.game_over;
  state.moves = data.moves || [];
  state.snapshots = data.snapshots || [];
  state.snapshotIndex = state.snapshots.length ? state.snapshots.length - 1 : 0;
  state.selected = null;
  state.legalTargets = [];
  if (data.last_user_move && state.context.length >= prevContextLen) {{
    pushLog('INFO', `User move ${{data.last_user_move.san || data.last_user_move.uci}}`);
  }}
  if (data.last_model_move) {{
    const modelMove = data.last_model_move;
    if (modelMove.error) {{
      const level = /legal/i.test(modelMove.error) ? 'ERROR' : 'INFO';
      const suffix = modelMove.fallback ? ' (fallback applied)' : '';
      pushLog(level, `Model: ${{modelMove.error}}${{suffix}}`);
    }}
    if (modelMove.uci) {{
      const tag = modelMove.fallback ? ' [fallback]' : '';
      pushLog('INFO', `Model move ${{modelMove.san || modelMove.uci}}${{tag}}`);
    }}
  }}
  renderAll();
}}

function buildSnapshots() {{
  // Server returns snapshots in /api/state and /api/move responses.
  return state.snapshots || [];
}}

function renderBoard() {{
  const snap = buildSnapshots()[state.snapshotIndex] || {{ fen: '8/8/8/8/8/8/8/8 w - - 0 1' }};
  const ranks = fenBoard(snap.fen);
  let html = '';
  for (let r = 0; r < 8; r++) {{
    const rankNum = 8 - r;
    const cells = expandRank(ranks[r]);
    for (let c = 0; c < 8; c++) {{
      const sq = squareName(r,c);
      const sqColor = (r + c) % 2 === 0 ? 'light' : 'dark';
      const piece = cells[c];
      const showRank = c === 0;
      const showFile = r === 7;
      const classes = ['square', sqColor];
      if (sameSquare(state.selected, sq) && state.snapshotIndex === state.snapshots.length - 1) classes.push('sel');
      if (state.legalTargets.includes(sq) && state.snapshotIndex === state.snapshots.length - 1) classes.push('hint');
      html += `<div class=\"${{classes.join(' ')}}\" data-square=\"${{sq}}\">`;
      if (piece) html += `<img class=\"piece\" alt=\"${{piece}}\" src=\"${{pieceToAsset(piece)}}\">`;
      if (showRank) html += `<span class=\"coord rank\">${{rankNum}}</span>`;
      if (showFile) html += `<span class=\"coord file\">${{files[c]}}</span>`;
      html += `</div>`;
    }}
  }}
  boardEl.innerHTML = html;
  boardEl.querySelectorAll('[data-square]').forEach((el) => el.addEventListener('click', onSquareClick));
}}

function renderMoves() {{
  const rows = [];
  for (let i = 0; i < state.moves.length; i += 2) {{
    const w = state.moves[i]; const b = state.moves[i+1];
    const no = Math.floor(i/2) + 1;
    const activePly = buildSnapshots()[state.snapshotIndex]?.ply || 0;
    const wActive = w && w.ply === activePly;
    const bActive = b && b.ply === activePly;
    rows.push(`<div class=\"move-row ${{(wActive||bActive)?'active':''}}\">`
      + `<div>${{no}}.</div>`
      + `<div data-jump=\"${{w ? w.ply : 0}}\">${{w ? w.san : ''}}</div>`
      + `<div data-jump=\"${{b ? b.ply : 0}}\">${{b ? b.san : ''}}</div>`
      + `</div>`);
  }}
  movesEl.innerHTML = rows.join('');
  movesEl.querySelectorAll('[data-jump]').forEach((el) => el.addEventListener('click', () => {{
    const ply = Number(el.getAttribute('data-jump')||'0');
    if (!Number.isNaN(ply)) {{ state.snapshotIndex = ply; renderAll(); }}
  }}));
}}

function renderStatus() {{
  const snap = buildSnapshots()[state.snapshotIndex] || {{ ply: 0 }};
  statusEl.textContent = `Ply ${{snap.ply || 0}} / ${{Math.max(0, (state.snapshots?.length||1)-1)}}`;
  turnTxt.textContent = state.turn;
  resultTxt.textContent = state.result;
  selectedTxt.textContent = state.selected || 'none';
  document.getElementById('prevBtn').disabled = state.snapshotIndex <= 0;
  document.getElementById('firstBtn').disabled = state.snapshotIndex <= 0;
  document.getElementById('nextBtn').disabled = state.snapshotIndex >= (state.snapshots.length - 1);
  document.getElementById('lastBtn').disabled = state.snapshotIndex >= (state.snapshots.length - 1);
  document.getElementById('undoBtn').disabled = state.context.length < 2;
}}

function renderLog() {{
  logPanelEl.classList.toggle('hidden', !state.logVisible);
  toggleLogBtn.textContent = state.logVisible ? 'Hide Log' : 'Show Log';
  if (!state.logs.length) {{
    logPanelEl.innerHTML = '<div class=\"small\">No log entries yet.</div>';
    return;
  }}
  logPanelEl.innerHTML = state.logs.slice().reverse().map((entry) =>
    `<div class=\"log-entry ${{entry.level === 'ERROR' ? 'error' : 'info'}}\">[${{entry.ts}}] ${{entry.level}} ${{entry.message}}</div>`
  ).join('');
}}

function renderAll() {{ renderBoard(); renderMoves(); renderStatus(); renderLog(); }}

function currentBoardPieceMap() {{
  const snap = buildSnapshots()[state.snapshots.length - 1];
  if (!snap) return {{}};
  const ranks = fenBoard(snap.fen);
  const out = {{}};
  for (let r=0; r<8; r++) {{
    const cells = expandRank(ranks[r]);
    for (let c=0; c<8; c++) {{ const p = cells[c]; if (p) out[squareName(r,c)] = p; }}
  }}
  return out;
}}

function guessLegalTargets(fromSq) {{
  // Visual-only hints. Server remains source of truth for legality.
  const map = currentBoardPieceMap();
  const p = map[fromSq];
  if (!p) return [];
  const isWhite = p === p.toUpperCase();
  const sideToMoveWhite = state.turn === 'white';
  if (isWhite !== sideToMoveWhite) return [];
  return [];
}}

async function onSquareClick(e) {{
  if (state.snapshotIndex !== state.snapshots.length - 1) return;
  if (state.game_over) return;
  const sq = e.currentTarget.getAttribute('data-square');
  if (!state.selected) {{
    state.selected = sq;
    state.legalTargets = guessLegalTargets(sq);
    renderAll();
    return;
  }}
  if (state.selected === sq) {{
    state.selected = null; state.legalTargets = []; renderAll(); return;
  }}
  const uci = state.selected + sq;
  setError('');
  try {{
    const data = await api('/api/move', {{
      context: state.context,
      user_move: uci,
      winner_side: winnerSideSel.value,
      topk: DEFAULTS.topk,
      user_color: 'white'
    }});
    applyServerState(data);
  }} catch (err) {{
    setError(err.message);
    state.selected = null;
    state.legalTargets = [];
    renderAll();
  }}
}}

async function refreshState() {{
  const data = await api('/api/state', {{ context: state.context }});
  applyServerState(data);
}}

document.getElementById('newBtn').addEventListener('click', async () => {{ state.context = []; setError(''); await refreshState(); }});
document.getElementById('undoBtn').addEventListener('click', async () => {{
  state.context = state.context.slice(0, Math.max(0, state.context.length - 2));
  setError('');
  await refreshState();
}});
document.getElementById('firstBtn').addEventListener('click', () => {{ state.snapshotIndex = 0; renderAll(); }});
document.getElementById('prevBtn').addEventListener('click', () => {{ state.snapshotIndex = Math.max(0, state.snapshotIndex - 1); renderAll(); }});
document.getElementById('nextBtn').addEventListener('click', () => {{ state.snapshotIndex = Math.min(state.snapshots.length - 1, state.snapshotIndex + 1); renderAll(); }});
document.getElementById('lastBtn').addEventListener('click', () => {{ state.snapshotIndex = state.snapshots.length - 1; renderAll(); }});
toggleLogBtn.addEventListener('click', () => {{ state.logVisible = !state.logVisible; renderLog(); }});
window.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowLeft') {{ state.snapshotIndex = Math.max(0, state.snapshotIndex - 1); renderAll(); }}
  if (e.key === 'ArrowRight') {{ state.snapshotIndex = Math.min(state.snapshots.length - 1, state.snapshotIndex + 1); renderAll(); }}
  if (e.key === 'Home') {{ state.snapshotIndex = 0; renderAll(); }}
  if (e.key === 'End') {{ state.snapshotIndex = state.snapshots.length - 1; renderAll(); }}
}});

(async function init() {{
  try {{ await refreshState(); }} catch (err) {{ setError(err.message); }}
}})();
</script>
</body></html>
"""


def snapshots_from_context(context: List[str]) -> List[Dict]:
    board = chess.Board()
    snaps = [{"ply": 0, "fen": board.fen()}]
    for ply, uci in enumerate(context, start=1):
        mv = chess.Move.from_uci(uci)
        board.push(mv)
        snaps.append({"ply": ply, "fen": board.fen()})
    return snaps


def state_response(context: List[str]) -> Dict:
    state = serialize_state(context)
    state["snapshots"] = snapshots_from_context(context)
    return state


def move_response(model_runtime: LoadedMoveModel, context: List[str], user_move: str, cfg: PlayConfig) -> Dict:
    state = apply_user_and_model_move(model_runtime, context, user_move, cfg)
    state["snapshots"] = snapshots_from_context(state["context"])
    return state
