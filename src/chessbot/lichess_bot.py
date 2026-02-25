from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, TextIO

import chess
from src.chessbot.io_utils import ensure_parent, write_json
from src.chessbot.play_vs_model import LoadedMoveModel
from src.chessbot.validation import game_id_for, winner_from_result


LICHESS_BASE = "https://lichess.org"
DEFAULT_KEYRING_SERVICE = "lichess"
DEFAULT_KEYRING_USERNAME = "lichess_api_token"
DEFAULT_PLAYED_GAMES_OUT = "data/live_play/lichess_bot_archive/valid_games.jsonl"


@dataclass
class BotConfig:
    token: str
    model_path: str
    winner_side: str = "W"
    topk: int = 10
    user_agent: str = "chessbot-lichess/0.1"
    accept_rated: bool = False
    allow_variants: tuple = ("standard",)
    min_initial_seconds: int = 0
    max_games: int = 1
    dry_run: bool = False
    preview_live_dir: str = ""
    self_user_id: str = ""
    played_games_out: str = DEFAULT_PLAYED_GAMES_OUT


@dataclass
class BotDecision:
    move_uci: str
    topk: List[str] = field(default_factory=list)
    predicted_uci: str = ""
    fallback: bool = False
    error: str = ""


class MoveProvider:
    def choose(self, context: List[str], board: chess.Board, winner_side: str, topk: int) -> BotDecision:
        raise NotImplementedError


class ModelMoveProvider(MoveProvider):
    def __init__(self, model_path: str):
        self.model = LoadedMoveModel.from_path(model_path)

    def choose(self, context: List[str], board: chess.Board, winner_side: str, topk: int) -> BotDecision:
        infer = self.model.infer(context=context, winner_side=winner_side, topk=topk)
        topk_tokens = [t for t in infer.get("topk", []) if isinstance(t, str)]
        predicted = topk_tokens[0] if topk_tokens else ""
        legal_uci = infer.get("best_legal", "") or ""
        if legal_uci:
            try:
                mv = chess.Move.from_uci(legal_uci)
            except Exception:
                mv = None
            if mv is not None and mv in board.legal_moves:
                return BotDecision(move_uci=legal_uci, topk=topk_tokens, predicted_uci=predicted)
        fallback_mv = next(iter(board.legal_moves), None)
        if fallback_mv is None:
            return BotDecision(move_uci="", topk=topk_tokens, predicted_uci=predicted, fallback=True, error="no legal moves")
        return BotDecision(
            move_uci=fallback_mv.uci(),
            topk=topk_tokens,
            predicted_uci=predicted,
            fallback=True,
            error="model fallback used",
        )


class LichessTransport:
    def stream_incoming_events(self) -> Iterable[Dict]:
        raise NotImplementedError

    def stream_game_state(self, game_id: str) -> Iterable[Dict]:
        raise NotImplementedError

    def accept_challenge(self, challenge_id: str) -> None:
        raise NotImplementedError

    def decline_challenge(self, challenge_id: str, reason: str) -> None:
        raise NotImplementedError

    def make_move(self, game_id: str, move_uci: str) -> None:
        raise NotImplementedError

    def create_challenge(
        self,
        username: str,
        *,
        rated: bool = False,
        clock_limit: int = 300,
        clock_increment: int = 0,
        color: str = "random",
        variant: str = "standard",
    ) -> Dict:
        raise NotImplementedError


class LichessHTTPTransport(LichessTransport):
    def __init__(
        self,
        token: str,
        user_agent: str,
        base_url: str = LICHESS_BASE,
        timeout_s: int = 30,
        min_request_interval_s: float = 1.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.min_request_interval_s = max(0.0, float(min_request_interval_s))
        self._last_request_started = 0.0
        self.headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": user_agent,
            "Accept": "application/x-ndjson, application/json",
        }

    def _request(self, method: str, path: str, data: Optional[bytes] = None) -> urllib.request.addinfourl:
        if self.min_request_interval_s > 0:
            now = time.monotonic()
            wait_s = self.min_request_interval_s - (now - self._last_request_started)
            if wait_s > 0:
                time.sleep(wait_s)
            self._last_request_started = time.monotonic()
        req = urllib.request.Request(self.base_url + path, data=data, method=method, headers=self.headers)
        return urllib.request.urlopen(req, timeout=self.timeout_s)

    def _iter_ndjson(self, response) -> Iterator[Dict]:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj

    def stream_incoming_events(self) -> Iterable[Dict]:
        with self._request("GET", "/api/stream/event") as resp:
            yield from self._iter_ndjson(resp)

    def stream_game_state(self, game_id: str) -> Iterable[Dict]:
        with self._request("GET", f"/api/bot/game/stream/{urllib.parse.quote(game_id)}") as resp:
            yield from self._iter_ndjson(resp)

    def accept_challenge(self, challenge_id: str) -> None:
        with self._request("POST", f"/api/challenge/{urllib.parse.quote(challenge_id)}/accept", data=b""):
            return None

    def decline_challenge(self, challenge_id: str, reason: str) -> None:
        body = urllib.parse.urlencode({"reason": reason}).encode("utf-8")
        with self._request("POST", f"/api/challenge/{urllib.parse.quote(challenge_id)}/decline", data=body):
            return None

    def make_move(self, game_id: str, move_uci: str) -> None:
        with self._request("POST", f"/api/bot/game/{urllib.parse.quote(game_id)}/move/{move_uci}", data=b""):
            return None

    def account(self) -> Dict:
        with self._request("GET", "/api/account") as resp:
            try:
                payload = json.load(resp)
            except Exception:
                return {}
        return payload if isinstance(payload, dict) else {}

    def create_challenge(
        self,
        username: str,
        *,
        rated: bool = False,
        clock_limit: int = 300,
        clock_increment: int = 0,
        color: str = "random",
        variant: str = "standard",
    ) -> Dict:
        body = urllib.parse.urlencode(
            {
                "rated": "true" if rated else "false",
                "clock.limit": str(max(0, int(clock_limit))),
                "clock.increment": str(max(0, int(clock_increment))),
                "color": str(color or "random"),
                "variant": str(variant or "standard"),
            }
        ).encode("utf-8")
        with self._request("POST", f"/api/challenge/{urllib.parse.quote(username)}", data=body) as resp:
            try:
                payload = json.load(resp)
            except Exception:
                return {"ok": False, "error": "invalid_json_response"}
        return payload if isinstance(payload, dict) else {"ok": False, "response": payload}


def board_from_moves_text(moves_text: str) -> tuple[chess.Board, List[str]]:
    board = chess.Board()
    context: List[str] = []
    for idx, uci in enumerate([m for m in moves_text.split() if m], start=1):
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            raise ValueError(f"Illegal move in game stream at ply {idx}: {uci}")
        board.push(mv)
        context.append(uci)
    return board, context


def _challenge_decline_reason(challenge: Dict, cfg: BotConfig) -> str:
    variant = ((challenge.get("variant") or {}).get("key") or "standard").lower()
    rated = bool(challenge.get("rated"))
    tc = challenge.get("timeControl") or {}
    initial = int(tc.get("limit", 0) or 0)
    if variant not in cfg.allow_variants:
        return "standard"
    if rated and not cfg.accept_rated:
        return "casual"
    if initial < cfg.min_initial_seconds:
        return "tooFast"
    return ""


def _result_from_live_state(latest_state: Dict, board: chess.Board) -> str:
    winner = str((latest_state or {}).get("winner", "") or "").lower()
    if winner == "white":
        return "1-0"
    if winner == "black":
        return "0-1"
    if board.is_game_over(claim_draw=True):
        return board.result(claim_draw=True)
    return "*"


def _valid_record_from_live_transcript(game_id: str, transcript: List[Dict]) -> Optional[Dict]:
    latest_state: Dict = {}
    white: Dict = {}
    black: Dict = {}
    for evt in transcript:
        t = evt.get("type")
        if t == "gameFull":
            white = dict((evt.get("white") or {}))
            black = dict((evt.get("black") or {}))
            latest_state = dict((evt.get("state") or {}))
        elif t == "gameState":
            latest_state = dict(evt)

    moves_text = str((latest_state or {}).get("moves", "") or "").strip()
    if not moves_text:
        return None
    board, moves_uci = board_from_moves_text(moves_text)
    result = _result_from_live_state(latest_state, board)
    now_utc = time.gmtime()
    date_str = f"{now_utc.tm_year:04d}.{now_utc.tm_mon:02d}.{now_utc.tm_mday:02d}"

    headers = {
        "Event": "Lichess Bot Live",
        "Site": f"https://lichess.org/{game_id}",
        "Date": date_str,
        "Round": "-",
        "White": str(white.get("name") or white.get("id") or "white"),
        "Black": str(black.get("name") or black.get("id") or "black"),
        "Result": result,
        "LichessGameId": game_id,
    }
    if white.get("rating") is not None:
        headers["WhiteElo"] = str(white.get("rating"))
    if black.get("rating") is not None:
        headers["BlackElo"] = str(black.get("rating"))
    gid = game_id_for(headers, "lichess_live_bot", moves_uci)
    return {
        "game_id": gid,
        "source_file": "lichess_live_bot",
        "headers": headers,
        "result": result,
        "winner_side": winner_from_result(result),
        "plies": len(moves_uci),
        "moves_uci": moves_uci,
    }


class LiveGameArchive:
    def __init__(self, out_path: str):
        self.out_path = str(out_path or "").strip()
        self._seen_game_ids: set[str] = set()

    def archive_from_transcript(self, game_id: str, transcript: List[Dict]) -> Optional[Dict]:
        if not self.out_path or game_id in self._seen_game_ids:
            return None
        record = _valid_record_from_live_transcript(game_id, transcript)
        if not record:
            return None
        ensure_parent(self.out_path)
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
        self._seen_game_ids.add(game_id)
        return record


class LichessBotRunner:
    def __init__(
        self,
        cfg: BotConfig,
        transport: LichessTransport,
        move_provider: MoveProvider,
        log: Callable[[Dict], None],
        preview_store: Optional[LivePreviewStore] = None,
        live_game_archive: Optional[LiveGameArchive] = None,
    ):
        self.cfg = cfg
        self.transport = transport
        self.move_provider = move_provider
        self.log = log
        self.preview_store = preview_store
        self.live_game_archive = live_game_archive
        self.active_games = 0

    def handle_challenge(self, challenge: Dict) -> None:
        cid = challenge.get("id", "")
        direction = str(challenge.get("direction", "") or "").lower()
        challenger = challenge.get("challenger") or {}
        challenger_id = str((challenger.get("id") or "")).lower()
        self_id = str(self.cfg.self_user_id or "").lower()
        is_outbound = direction == "out" or bool(self_id and challenger_id and challenger_id == self_id)
        if is_outbound:
            self.log({"event": "challenge_outbound_seen", "id": cid})
            return
        reason = _challenge_decline_reason(challenge, self.cfg)
        if reason:
            self.transport.decline_challenge(cid, reason)
            self.log({"event": "challenge_declined", "id": cid, "reason": reason})
            return
        self.transport.accept_challenge(cid)
        self.log({"event": "challenge_accepted", "id": cid})

    def _maybe_play_turn(self, game_id: str, bot_color: chess.Color, state: Dict) -> Optional[Dict]:
        moves_text = state.get("moves", "") or ""
        board, context = board_from_moves_text(moves_text)
        if board.is_game_over(claim_draw=True):
            return None
        if board.turn != bot_color:
            return None
        decision = self.move_provider.choose(context=context, board=board, winner_side=self.cfg.winner_side, topk=self.cfg.topk)
        if not decision.move_uci:
            self.log({"event": "no_move", "game_id": game_id, "error": decision.error})
            return None
        if not self.cfg.dry_run:
            self.transport.make_move(game_id, decision.move_uci)
        payload = {
            "event": "move_played",
            "game_id": game_id,
            "move_uci": decision.move_uci,
            "fallback": decision.fallback,
            "predicted_uci": decision.predicted_uci,
            "topk": decision.topk,
        }
        if decision.error:
            payload["error"] = decision.error
        self.log(payload)
        return payload

    def play_game(self, game_id: str, bot_color_hint: Optional[chess.Color] = None) -> List[Dict]:
        transcript: List[Dict] = []
        bot_color: Optional[chess.Color] = bot_color_hint
        for evt in self.transport.stream_game_state(game_id):
            t = evt.get("type")
            transcript.append(evt)
            if self.preview_store is not None:
                self.preview_store.on_game_event(game_id, evt, transcript)
            try:
                if t == "gameFull":
                    white = (evt.get("white") or {})
                    black = (evt.get("black") or {})
                    if bot_color is not None:
                        pass
                    elif self.cfg.self_user_id:
                        self_id = str(self.cfg.self_user_id).lower()
                        if str(white.get("id", "") or "").lower() == self_id:
                            bot_color = chess.WHITE
                        elif str(black.get("id", "") or "").lower() == self_id:
                            bot_color = chess.BLACK
                    elif white.get("title") == "BOT" and black.get("title") == "BOT":
                        bot_color = bot_color_hint if bot_color_hint is not None else chess.WHITE
                    elif white.get("title") == "BOT":
                        bot_color = chess.WHITE
                    elif black.get("title") == "BOT":
                        bot_color = chess.BLACK
                    else:
                        bot_color = chess.WHITE
                    state = evt.get("state") or {}
                    if bot_color is not None:
                        self._maybe_play_turn(game_id, bot_color, state)
                elif t == "gameState" and bot_color is not None:
                    self._maybe_play_turn(game_id, bot_color, evt)
                elif t == "chatLine":
                    self.log(
                        {"event": "chat", "game_id": game_id, "username": evt.get("username", ""), "text": evt.get("text", "")}
                    )
            except Exception as exc:
                self.log(
                    {
                        "event": "game_event_error",
                        "game_id": game_id,
                        "stream_event_type": t,
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=5),
                    }
                )
                raise
        self.log({"event": "game_stream_ended", "game_id": game_id})
        if self.live_game_archive is not None:
            try:
                archived = self.live_game_archive.archive_from_transcript(game_id, transcript)
                if archived is not None:
                    self.log(
                        {
                            "event": "live_game_archived",
                            "game_id": game_id,
                            "out_path": self.live_game_archive.out_path,
                            "plies": int(archived.get("plies", 0)),
                            "result": str(archived.get("result", "*")),
                        }
                    )
            except Exception as exc:
                self.log(
                    {
                        "event": "live_game_archive_error",
                        "game_id": game_id,
                        "error": str(exc),
                    }
                )
        return transcript

    def run(self) -> None:
        for evt in self.transport.stream_incoming_events():
            t = evt.get("type")
            try:
                if t == "challenge":
                    self.handle_challenge(evt.get("challenge") or {})
                elif t == "gameStart":
                    game_obj = evt.get("game") or {}
                    gid = game_obj.get("id", "")
                    if gid:
                        self.log({"event": "game_start", "game_id": gid})
                        color_hint = None
                        color_s = str(game_obj.get("color", "") or "").lower()
                        if color_s == "white":
                            color_hint = chess.WHITE
                        elif color_s == "black":
                            color_hint = chess.BLACK
                        self.play_game(gid, bot_color_hint=color_hint)
                else:
                    self.log({"event": "event_stream_item", "stream_event_type": t or "", "payload": evt})
            except Exception as exc:
                self.log(
                    {
                        "event": "runner_event_error",
                        "stream_event_type": t or "",
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=5),
                    }
                )


def _piece_asset_relpath(piece: chess.Piece) -> str:
    letter = piece.symbol().upper()
    color = "w" if piece.color == chess.WHITE else "b"
    return f"{color}{letter}.svg"


def _move_rows_from_moves_text(moves_text: str) -> List[Dict]:
    board = chess.Board()
    rows: List[Dict] = []
    for ply, uci in enumerate([m for m in (moves_text or "").split() if m], start=1):
        mv = chess.Move.from_uci(uci)
        san = board.san(mv)
        board.push(mv)
        rows.append({"ply": ply, "uci": uci, "san": san})
    return rows


def render_preview_html(
    game_id: str,
    transcript: List[Dict],
    logs: List[Dict],
    latest_state: Optional[Dict] = None,
    piece_base: str = "",
) -> str:
    rows = []
    for i, item in enumerate(logs, start=1):
        rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{html.escape(str(item.get('event', '')))}</td>"
            f"<td><code>{html.escape(json.dumps(item, ensure_ascii=True, sort_keys=True))}</code></td>"
            "</tr>"
        )
    latest_state = dict(latest_state or {})
    moves_text = str(latest_state.get("moves", "") or "")
    board = None
    board_error = ""
    move_rows: List[Dict] = []
    if latest_state:
        try:
            board, _ = board_from_moves_text(moves_text)
            move_rows = _move_rows_from_moves_text(moves_text)
        except Exception as exc:
            board_error = str(exc)
    board_squares = []
    if board is not None:
        for rank in range(8, 0, -1):
            for file_idx, file_ch in enumerate("abcdefgh"):
                sq = chess.parse_square(f"{file_ch}{rank}")
                piece = board.piece_at(sq)
                is_light = (rank + file_idx) % 2 == 0
                piece_html = ""
                if piece and piece_base:
                    src = f"{piece_base.rstrip('/')}/{_piece_asset_relpath(piece)}"
                    alt = html.escape(piece.symbol())
                    piece_html = f'<img class="piece" src="{html.escape(src)}" alt="{alt}" />'
                elif piece:
                    piece_html = f'<span class="piece-txt">{html.escape(piece.symbol())}</span>'
                coord_html = ""
                if rank == 1:
                    coord_html += f'<span class="coord file">{file_ch}</span>'
                if file_idx == 0:
                    coord_html += f'<span class="coord rank">{rank}</span>'
                board_squares.append(
                    f'<div class="square {"light" if is_light else "dark"}">{piece_html}{coord_html}</div>'
                )
    move_rows_html = []
    for i in range(0, len(move_rows), 2):
        white = move_rows[i]
        black = move_rows[i + 1] if i + 1 < len(move_rows) else None
        move_no = (white["ply"] + 1) // 2
        move_rows_html.append(
            "<div class=\"move-row\">"
            f"<div>{move_no}.</div>"
            f"<div title=\"{html.escape(white['uci'])}\">{html.escape(white['san'])}</div>"
            f"<div title=\"{html.escape(black['uci']) if black else ''}\">{html.escape(black['san']) if black else ''}</div>"
            "</div>"
        )
    state_json = html.escape(json.dumps(latest_state, indent=2, ensure_ascii=True))
    transcript_json = html.escape(json.dumps(transcript, indent=2, ensure_ascii=True))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta http-equiv="refresh" content="2" />
<title>Lichess Bot Preview {html.escape(game_id)}</title>
<style>
body {{ margin:0; font-family: Georgia, serif; background: #f3efe6; color:#221b15; }}
.wrap {{ max-width: 1200px; margin: 24px auto; padding: 0 16px; display:grid; gap:16px; }}
.card {{ background:#fffaf1; border:1px solid #d7cbb6; border-radius:14px; padding:14px; box-shadow: 0 8px 24px rgba(20,14,8,.06); }}
table {{ width:100%; border-collapse: collapse; font-family: ui-monospace, monospace; font-size:12px; }}
th, td {{ border-bottom:1px solid #e9dfcf; text-align:left; padding:6px; vertical-align:top; }}
pre {{ margin:0; white-space:pre-wrap; word-break:break-word; font-family: ui-monospace, monospace; font-size:12px; }}
code {{ white-space:pre-wrap; word-break:break-word; }}
.split {{ display:grid; grid-template-columns: minmax(280px,540px) minmax(280px,1fr); gap:16px; align-items:start; }}
@media (max-width: 900px) {{ .split {{ grid-template-columns: 1fr; }} }}
.board {{
  width:min(100%, 520px); aspect-ratio:1; display:grid; grid-template-columns:repeat(8,1fr); grid-template-rows:repeat(8,1fr);
  border:2px solid #6d5234; border-radius:10px; overflow:hidden; margin:0 auto;
}}
.square {{ position:relative; aspect-ratio:1/1; }}
.square.light {{ background:#f0d9b5; }} .square.dark {{ background:#b58863; }}
.piece {{ width:100%; height:100%; display:block; padding:4%; }}
.piece-txt {{ display:grid; place-items:center; width:100%; height:100%; font:700 18px/1 ui-monospace, monospace; }}
.coord {{ position:absolute; font-size:11px; opacity:.7; font-weight:700; }}
.coord.file {{ right:4px; bottom:2px; }} .coord.rank {{ left:4px; top:2px; }}
.kv {{ display:grid; grid-template-columns: 90px 1fr; gap:6px 8px; font-size:14px; }}
.moves {{ max-height: 320px; overflow:auto; display:grid; gap:4px; font-family: ui-monospace, monospace; font-size:13px; }}
.move-row {{ display:grid; grid-template-columns:42px 1fr 1fr; gap:8px; padding:4px 6px; border-radius:8px; background:#fffdf7; border:1px solid #ebdfce; }}
.warn {{ color:#8a2d22; font-weight:600; }}
</style>
</head>
<body>
<div class="wrap">
  <section class="card"><h1 style="margin:0 0 8px;">Lichess Bot Preview</h1><div>game_id: <code>{html.escape(game_id)}</code></div></section>
  <section class="card split">
    <div>
      <h2 style="margin:0 0 8px;">Board</h2>
      <div class="board">{''.join(board_squares) if board_squares else '<div class="warn">No board state yet</div>'}</div>
      {f'<div class="warn" style="margin-top:8px;">{html.escape(board_error)}</div>' if board_error else ''}
    </div>
    <div>
      <h2 style="margin:0 0 8px;">Game State</h2>
      <div class="kv">
        <b>Status</b><span>{html.escape(str(latest_state.get("status", "")))}</span>
        <b>Moves</b><span>{len(move_rows)}</span>
        <b>Turn</b><span>{'white' if (board is not None and board.turn == chess.WHITE) else ('black' if board is not None else '')}</span>
        <b>Result</b><span>{html.escape(board.result(claim_draw=True) if board is not None and board.is_game_over(claim_draw=True) else '*')}</span>
        <b>Last Move</b><span><code>{html.escape(moves_text.split()[-1] if moves_text.strip() else '')}</code></span>
      </div>
      <h3 style="margin:12px 0 8px;">Move List</h3>
      <div class="moves">{''.join(move_rows_html) or '<div>No moves yet</div>'}</div>
    </div>
  </section>
  <section class="card"><h2 style="margin:0 0 8px;">Bot Actions</h2><table><thead><tr><th>#</th><th>event</th><th>payload</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
  <section class="card"><h2 style="margin:0 0 8px;">Latest State JSON</h2><pre>{state_json}</pre></section>
  <section class="card"><h2 style="margin:0 0 8px;">Game Stream Transcript</h2><pre>{transcript_json}</pre></section>
</div>
</body>
</html>"""


def render_live_index_html(index_obj: Dict) -> str:
    games = index_obj.get("games") or []
    rows = []
    for item in games:
        gid = str(item.get("game_id", ""))
        status = html.escape(str(item.get("status", "")))
        updated = html.escape(str(item.get("updated_at_epoch", "")))
        last_move = html.escape(str(item.get("last_move", "")))
        rows.append(
            "<tr>"
            f"<td><a href=\"games/{html.escape(gid)}/index.html\">{html.escape(gid)}</a></td>"
            f"<td>{status}</td>"
            f"<td><code>{last_move}</code></td>"
            f"<td>{updated}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta http-equiv="refresh" content="2" />
<title>Lichess Bot Live Preview</title>
<style>
body {{ margin:0; font-family: Georgia, serif; background:#f4efe4; color:#211a14; }}
.wrap {{ max-width: 1100px; margin: 24px auto; padding: 0 16px; display:grid; gap:16px; }}
.card {{ background:#fffaf0; border:1px solid #d8cbb5; border-radius:14px; padding:14px; }}
table {{ width:100%; border-collapse: collapse; font-family: ui-monospace, monospace; font-size:13px; }}
th, td {{ border-bottom:1px solid #eadfce; text-align:left; padding:8px 6px; }}
code {{ white-space:pre-wrap; }}
</style>
</head>
<body>
<div class="wrap">
  <section class="card">
    <h1 style="margin:0 0 8px;">Lichess Bot Live Preview</h1>
    <div>Updated: <code>{html.escape(str(index_obj.get("updated_at_epoch", "")))}</code></div>
    <div>Global log events: <code>{html.escape(str(index_obj.get("global_log_count", 0)))}</code></div>
  </section>
  <section class="card">
    <h2 style="margin:0 0 8px;">Games</h2>
    <table>
      <thead><tr><th>Game</th><th>Status</th><th>Last Move</th><th>Updated</th></tr></thead>
      <tbody>{''.join(rows) or '<tr><td colspan="4">No games yet</td></tr>'}</tbody>
    </table>
  </section>
  <section class="card">
    <h2 style="margin:0 0 8px;">Files</h2>
    <div><a href="index.json">index.json</a></div>
    <div><a href="logs.json">logs.json</a></div>
  </section>
</div>
</body>
</html>"""


class LivePreviewStore:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.global_logs: List[Dict] = []
        self.games: Dict[str, Dict] = {}
        self._ensure_piece_assets_link()
        self._write_global()

    def _ensure_piece_assets_link(self) -> None:
        target = Path(__file__).resolve().parents[2] / "assets" / "pieces" / "cburnett"
        if not target.exists():
            return
        link_dir = self.base_dir / "assets" / "pieces"
        link_dir.mkdir(parents=True, exist_ok=True)
        link_path = link_dir / "cburnett"
        if link_path.exists() or link_path.is_symlink():
            return
        try:
            link_path.symlink_to(target, target_is_directory=True)
        except Exception:
            # Fallback to no symlink; preview will still render text pieces.
            return

    def _game_dir(self, game_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in game_id) or "unknown"
        path = self.base_dir / "games" / safe
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_global(self) -> None:
        write_json(str(self.base_dir / "logs.json"), {"logs": self.global_logs[-1000:]})
        games_index = []
        for gid, g in sorted(self.games.items(), key=lambda kv: kv[1].get("updated_at_epoch", 0), reverse=True):
            latest_state = g.get("latest_state") or {}
            moves_text = str(latest_state.get("moves", "") or "")
            games_index.append(
                {
                    "game_id": gid,
                    "updated_at_epoch": g.get("updated_at_epoch", 0),
                    "status": latest_state.get("status", ""),
                    "last_move": (moves_text.split()[-1] if moves_text.strip() else ""),
                    "transcript_count": len(g.get("transcript", [])),
                    "action_log_count": len(g.get("logs", [])),
                }
            )
        index_obj = {
            "updated_at_epoch": time.time(),
            "global_log_count": len(self.global_logs),
            "games": games_index,
        }
        write_json(str(self.base_dir / "index.json"), index_obj)

    def on_log(self, payload: Dict) -> None:
        item = dict(payload)
        item.setdefault("ts_epoch", time.time())
        self.global_logs.append(item)
        gid = item.get("game_id")
        if gid:
            game = self.games.setdefault(str(gid), {"transcript": [], "logs": [], "latest_state": {}})
            game["logs"].append(item)
            game["updated_at_epoch"] = item["ts_epoch"]
            self._write_game(str(gid))
        self._write_global()

    def on_game_event(self, game_id: str, evt: Dict, transcript: List[Dict]) -> None:
        game = self.games.setdefault(str(game_id), {"transcript": [], "logs": [], "latest_state": {}})
        game["transcript"] = [dict(x) for x in transcript]
        game["updated_at_epoch"] = time.time()
        t = evt.get("type")
        if t == "gameFull":
            state = dict((evt.get("state") or {}))
            state["type"] = "gameFull.state"
            game["latest_state"] = state
            game["players"] = {
                "white": dict((evt.get("white") or {})),
                "black": dict((evt.get("black") or {})),
            }
        elif t == "gameState":
            game["latest_state"] = dict(evt)
        self._write_game(str(game_id))
        self._write_global()

    def _write_game(self, game_id: str) -> None:
        game = self.games.setdefault(str(game_id), {"transcript": [], "logs": [], "latest_state": {}})
        gdir = self._game_dir(game_id)
        latest_state = game.get("latest_state", {}) or {}
        state_payload = {"game_id": game_id, "state": latest_state}
        players = game.get("players")
        if isinstance(players, dict):
            state_payload["players"] = players
        moves_text = str(latest_state.get("moves", "") or "")
        if moves_text:
            try:
                board, _ = board_from_moves_text(moves_text)
                state_payload["derived"] = {
                    "fen": board.fen(),
                    "turn": "white" if board.turn == chess.WHITE else "black",
                    "is_game_over": bool(board.is_game_over(claim_draw=True)),
                    "result": board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*",
                    "move_rows": _move_rows_from_moves_text(moves_text),
                }
            except Exception as exc:
                state_payload["derived_error"] = str(exc)
        write_json(str(gdir / "state.json"), state_payload)
        write_json(str(gdir / "actions.json"), {"game_id": game_id, "logs": game.get("logs", [])})
        write_json(str(gdir / "transcript.json"), {"game_id": game_id, "transcript": game.get("transcript", [])})


class _JsonLogger:
    def __init__(
        self,
        out: TextIO,
        sinks: Optional[List[Callable[[Dict], None]]] = None,
        jsonl_path: str = "",
    ):
        self.out = out
        self.records: List[Dict] = []
        self.sinks = sinks or []
        self.jsonl_path = str(jsonl_path or "")

    def __call__(self, payload: Dict) -> None:
        item = dict(payload)
        item.setdefault("ts_epoch", time.time())
        self.records.append(item)
        line = json.dumps(item, ensure_ascii=True)
        self.out.write(line + "\n")
        self.out.flush()
        if self.jsonl_path:
            try:
                ensure_parent(self.jsonl_path)
                with open(self.jsonl_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                # Keep file logging best-effort.
                pass
        for sink in self.sinks:
            try:
                sink(item)
            except Exception:
                # Keep logging non-fatal; preview persistence issues should not kill live play.
                continue


def _find_latest_model(repo_root: Path) -> Path:
    candidates = [p for p in (repo_root / "artifacts").rglob("*.pt") if p.is_file()]
    if not candidates:
        raise SystemExit("No model artifacts found under artifacts/")
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p)))


def _preview_from_fixture(cfg: BotConfig, move_provider: MoveProvider, fixture_path: Path, preview_html_path: Optional[Path]) -> None:
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    game_id = str(data.get("game_id", "preview"))
    transcript = data.get("transcript") or []
    if not isinstance(transcript, list):
        raise SystemExit("Fixture transcript must be a list of Lichess NDJSON objects")

    class _FixtureTransport(LichessTransport):
        def __init__(self) -> None:
            self.moves: List[str] = []

        def stream_incoming_events(self) -> Iterable[Dict]:
            return []

        def stream_game_state(self, game_id: str) -> Iterable[Dict]:
            return transcript

        def accept_challenge(self, challenge_id: str) -> None:
            return None

        def decline_challenge(self, challenge_id: str, reason: str) -> None:
            return None

        def make_move(self, game_id: str, move_uci: str) -> None:
            self.moves.append(move_uci)

    logger = _JsonLogger(sys.stdout)
    runner = LichessBotRunner(cfg=cfg, transport=_FixtureTransport(), move_provider=move_provider, log=logger)
    runner.play_game(game_id)
    if preview_html_path:
        preview_html_path.parent.mkdir(parents=True, exist_ok=True)
        preview_html_path.write_text(render_preview_html(game_id, transcript, logger.records), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lichess bot play loop using local move model")
    p.add_argument("--token", default="", help="Lichess API token override (otherwise keyring, then LICHESS_BOT_TOKEN)")
    p.add_argument("--keyring-service", default=DEFAULT_KEYRING_SERVICE, help="Keyring service name for token lookup")
    p.add_argument("--keyring-username", default=DEFAULT_KEYRING_USERNAME, help="Keyring username for token lookup")
    p.add_argument("--model", default="latest", help="Model artifact path or 'latest'")
    p.add_argument("--winner-side", default="W", choices=["W", "B", "D", "?"], help="Model conditioning token")
    p.add_argument("--topk", type=int, default=10, help="Top-k predictions to search for a legal move")
    p.add_argument("--accept-rated", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--variants", default="standard", help="Comma-separated allowed variants")
    p.add_argument("--min-initial-seconds", type=int, default=0, help="Decline games faster than this initial clock")
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False, help="Do not POST moves to Lichess")
    p.add_argument(
        "--preview-live-dir",
        default="",
        help="Directory to persist live per-game JSON preview artifacts while bot runs",
    )
    p.add_argument(
        "--min-request-interval-ms",
        type=int,
        default=1200,
        help="Minimum delay between Lichess API requests in ms (conservative default)",
    )
    p.add_argument("--preview-fixture", default="", help="Local JSON fixture with game stream transcript for offline preview")
    p.add_argument("--preview-html", default="", help="Optional HTML output path for preview run")
    p.add_argument("--log-jsonl", default="", help="Optional append-only JSONL file for bot logs")
    p.add_argument(
        "--played-games-out",
        default=DEFAULT_PLAYED_GAMES_OUT,
        help="Append live-played games in validated JSONL schema format to this path (empty disables)",
    )
    p.add_argument("--challenge-user", default="", help="Create an outbound challenge to this Lichess username and exit")
    p.add_argument("--challenge-clock-limit", type=int, default=300, help="Outbound challenge initial clock in seconds")
    p.add_argument("--challenge-clock-increment", type=int, default=0, help="Outbound challenge increment in seconds")
    p.add_argument(
        "--challenge-color",
        default="random",
        choices=["white", "black", "random"],
        help="Preferred color for outbound challenge",
    )
    p.add_argument("--challenge-variant", default="standard", help="Variant for outbound challenge (default standard)")
    p.add_argument(
        "--challenge-rated",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Create rated outbound challenge",
    )
    return p


def _token_from_keyring(service: str, username: str) -> str:
    try:
        import keyring  # type: ignore
    except Exception:
        return ""
    try:
        token = keyring.get_password(service, username)
    except Exception:
        return ""
    return token or ""


def _resolve_live_token(args: argparse.Namespace) -> str:
    token = str(getattr(args, "token", "") or "")
    if token:
        return token
    token = _token_from_keyring(str(args.keyring_service), str(args.keyring_username))
    if token:
        return token
    return str(os.environ.get("LICHESS_BOT_TOKEN", "") or "")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.challenge_user and args.preview_fixture:
        raise SystemExit("--challenge-user cannot be used with --preview-fixture")
    repo_root = Path(__file__).resolve().parents[2]
    token = ""
    if not args.preview_fixture:
        token = _resolve_live_token(args)
    if not token and not args.preview_fixture:
        raise SystemExit(
            "Missing token. Pass --token, install/configure keyring "
            f"({args.keyring_service}/{args.keyring_username}), or set LICHESS_BOT_TOKEN"
        )

    if args.challenge_user:
        transport = LichessHTTPTransport(
            token=token,
            user_agent="chessbot-lichess/0.1",
            min_request_interval_s=max(0.0, float(args.min_request_interval_ms) / 1000.0),
        )
        result = transport.create_challenge(
            args.challenge_user,
            rated=bool(args.challenge_rated),
            clock_limit=int(args.challenge_clock_limit),
            clock_increment=int(args.challenge_clock_increment),
            color=str(args.challenge_color),
            variant=str(args.challenge_variant),
        )
        print(
            json.dumps(
                {
                    "event": "challenge_create_result",
                    "opponent": str(args.challenge_user),
                    "rated": bool(args.challenge_rated),
                    "clock_limit": int(args.challenge_clock_limit),
                    "clock_increment": int(args.challenge_clock_increment),
                    "color": str(args.challenge_color),
                    "variant": str(args.challenge_variant),
                    "response": result,
                },
                ensure_ascii=True,
            )
        )
        return 0

    model_path = _find_latest_model(repo_root) if args.model == "latest" else Path(args.model).resolve()

    cfg = BotConfig(
        token=token,
        model_path=str(model_path),
        winner_side=args.winner_side,
        topk=max(1, int(args.topk)),
        accept_rated=bool(args.accept_rated),
        allow_variants=tuple(v.strip().lower() for v in args.variants.split(",") if v.strip()),
        min_initial_seconds=max(0, int(args.min_initial_seconds)),
        dry_run=bool(args.dry_run),
        preview_live_dir=str(args.preview_live_dir or ""),
        played_games_out=str(args.played_games_out or ""),
    )
    move_provider = ModelMoveProvider(str(model_path))

    if args.preview_fixture:
        _preview_from_fixture(
            cfg=cfg,
            move_provider=move_provider,
            fixture_path=Path(args.preview_fixture),
            preview_html_path=Path(args.preview_html) if args.preview_html else None,
        )
        return 0

    preview_store = LivePreviewStore(cfg.preview_live_dir) if cfg.preview_live_dir else None
    logger = _JsonLogger(
        sys.stdout,
        sinks=([preview_store.on_log] if preview_store is not None else None),
        jsonl_path=str(args.log_jsonl or ""),
    )
    transport = LichessHTTPTransport(
        token=cfg.token,
        user_agent=cfg.user_agent,
        min_request_interval_s=max(0.0, float(args.min_request_interval_ms) / 1000.0),
    )
    try:
        acct = transport.account()
        cfg.self_user_id = str(acct.get("id", "") or "").lower()
    except Exception:
        cfg.self_user_id = ""
    runner = LichessBotRunner(
        cfg=cfg,
        transport=transport,
        move_provider=move_provider,
        log=logger,
        preview_store=preview_store,
        live_game_archive=LiveGameArchive(cfg.played_games_out) if cfg.played_games_out else None,
    )
    logger(
        {
            "event": "bot_start",
            "model_path": str(model_path),
            "dry_run": cfg.dry_run,
            "preview_live_dir": cfg.preview_live_dir,
            "min_request_interval_ms": int(args.min_request_interval_ms),
            "variants": list(cfg.allow_variants),
            "accept_rated": cfg.accept_rated,
        }
    )
    backoff = 1.0
    while True:
        try:
            runner.run()
            backoff = 1.0
        except KeyboardInterrupt:
            return 0
        except urllib.error.URLError as exc:
            logger({"event": "network_error", "error": str(exc), "retry_s": backoff})
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)
        except Exception as exc:
            logger(
                {
                    "event": "unexpected_error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=10),
                    "retry_s": backoff,
                }
            )
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)


if __name__ == "__main__":
    raise SystemExit(main())
