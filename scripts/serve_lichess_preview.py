#!/usr/bin/env python3
import argparse
import html
import http.server
import json
import socketserver
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]


APP_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Lichess Bot Live Preview</title>
<style>
:root {
  --bg: #f4efe4;
  --card: #fffaf0;
  --line: #d8cbb5;
  --text: #211a14;
  --muted: #6d5b49;
  --accent: #2f6b4f;
  --danger: #8a2d22;
}
* { box-sizing: border-box; }
body { margin: 0; font-family: Georgia, serif; background: var(--bg); color: var(--text); }
.wrap { max-width: 1280px; margin: 20px auto; padding: 0 14px; display: grid; gap: 14px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 14px; }
.toolbar { display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: center; }
.toolbar label { display: inline-flex; gap: 6px; align-items: center; font-size: 14px; }
input[type="number"], input[type="text"], select {
  border: 1px solid var(--line);
  background: #fffef9;
  color: var(--text);
  border-radius: 8px;
  padding: 6px 8px;
}
button {
  border: 1px solid #bfae95;
  background: #f7efdf;
  color: var(--text);
  border-radius: 9px;
  padding: 7px 10px;
  cursor: pointer;
}
button:hover { background: #f2e6cf; }
.pill { display:inline-block; padding: 2px 8px; border-radius: 999px; border:1px solid var(--line); background:#fffdf8; font-size:12px; }
.muted { color: var(--muted); }
.warn { color: var(--danger); }
.split { display: grid; grid-template-columns: 340px 1fr; gap: 14px; align-items: start; }
@media (max-width: 980px) { .split { grid-template-columns: 1fr; } }
table { width: 100%; border-collapse: collapse; font-family: ui-monospace, monospace; font-size: 12px; }
th, td { border-bottom: 1px solid #eadfce; text-align: left; padding: 6px; vertical-align: top; }
tbody tr:hover { background: #fff8e8; }
tbody tr.active { background: #efe4c8; }
pre, code { font-family: ui-monospace, monospace; }
pre { margin: 0; white-space: pre-wrap; word-break: break-word; font-size: 12px; }
.panel-title { margin: 0 0 8px; font-size: 16px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
@media (max-width: 980px) { .grid2 { grid-template-columns: 1fr; } }
.logs { max-height: 260px; overflow: auto; border:1px solid #eadfce; border-radius: 10px; background:#fffdf7; padding: 8px; }
.bot-list { max-height: 280px; overflow: auto; border:1px solid #eadfce; border-radius: 10px; background:#fffdf7; padding: 8px; }
.board-wrap { display:grid; gap:8px; }
.board-meta { font-size:12px; color: var(--muted); display:flex; gap:10px; flex-wrap:wrap; }
.board-grid {
  width:min(100%, 420px);
  aspect-ratio: 1;
  display:grid;
  grid-template-columns: repeat(8, 1fr);
  grid-template-rows: repeat(8, 1fr);
  border:2px solid #6d5234;
  border-radius: 10px;
  overflow:hidden;
  background:#d7c3a7;
}
.sq { position:relative; aspect-ratio:1/1; }
.sq.light { background:#f0d9b5; }
.sq.dark { background:#b58863; }
.sq.light.last-from, .sq.light.last-to { background:#cdd26a; }
.sq.dark.last-from, .sq.dark.last-to { background:#aaa23b; }
.sq.last-to::after {
  content:"";
  position:absolute;
  inset:0;
  box-shadow: inset 0 0 0 3px rgba(38, 114, 53, 0.55);
  pointer-events:none;
}
.sq .coord { position:absolute; font-size:10px; font-weight:700; opacity:0.7; line-height:1; }
.sq .coord.file { right:3px; bottom:2px; }
.sq .coord.rank { left:3px; top:2px; }
.sq img.piece { width:100%; height:100%; display:block; padding:4%; object-fit:contain; }
.sq .piece-txt { width:100%; height:100%; display:grid; place-items:center; font:700 18px/1 ui-monospace, monospace; }
.move-table { max-height:220px; overflow:auto; border:1px solid #eadfce; border-radius:10px; background:#fffdf7; }
.move-table table { font-size:12px; }
.log-row { padding: 6px 0; border-bottom: 1px solid #f0e6d8; }
.log-row:last-child { border-bottom: none; }
.bot-row { display:grid; grid-template-columns: 1fr auto; gap:8px; align-items:center; padding:6px 0; border-bottom:1px solid #f0e6d8; }
.bot-row:last-child { border-bottom:none; }
.btn-mini { padding:4px 7px; font-size:12px; border-radius:8px; }
.btn-row { display:flex; gap:6px; }
.rowline { display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }
.evt { font-weight: 700; color: var(--accent); }
.small { font-size: 12px; }
a { color: #234f85; text-decoration: none; }
a:hover { text-decoration: underline; }
</style>
</head>
<body>
<div class="wrap">
  <section class="card">
    <div class="toolbar">
      <span class="pill">Live Preview</span>
      <label><input id="autoRefresh" type="checkbox" checked /> Auto refresh</label>
      <label>Interval (ms) <input id="intervalMs" type="number" min="250" step="250" value="1000" style="width:100px" /></label>
      <label><input id="followNewest" type="checkbox" checked /> Follow newest game</label>
      <label>Game <select id="gameSelect"><option value="">(none)</option></select></label>
      <button id="refreshBtn" type="button">Refresh now</button>
      <span id="statusLine" class="muted small">Waiting for preview data...</span>
    </div>
  </section>

  <section class="card">
    <div class="toolbar">
      <span class="pill">Challenge</span>
      <label>User <input id="challengeUser" type="text" placeholder="opponent_username" style="width:220px" /></label>
      <label>Limit <input id="challengeLimit" type="number" min="0" step="1" value="300" style="width:90px" /></label>
      <label>Inc <input id="challengeInc" type="number" min="0" step="1" value="0" style="width:80px" /></label>
      <label>Color
        <select id="challengeColor">
          <option value="random">random</option>
          <option value="white">white</option>
          <option value="black">black</option>
        </select>
      </label>
      <label>Variant <input id="challengeVariant" type="text" value="standard" style="width:120px" /></label>
      <label><input id="challengeRated" type="checkbox" /> Rated</label>
      <button id="challengeBtn" type="button">Send challenge</button>
      <span id="challengeStatus" class="muted small">Not sent</span>
    </div>
  </section>

  <section class="card">
    <div class="toolbar">
      <span class="pill">Opponent Search</span>
      <label><input id="opponentSearchEnabled" type="checkbox" /> Actively search</label>
      <label>Interval (ms) <input id="opponentSearchIntervalMs" type="number" min="1000" step="500" value="15000" style="width:110px" /></label>
      <button id="opponentSearchRefreshBtn" type="button">Refresh bots now</button>
      <span id="opponentSearchStatus" class="muted small">Search off</span>
    </div>
    <div class="toolbar" style="margin-top:10px;">
      <span class="pill">Auto Challenge</span>
      <label><input id="autoChallengeEnabled" type="checkbox" /> Continuously challenge</label>
      <label>Interval (ms) <input id="autoChallengeIntervalMs" type="number" min="2000" step="1000" value="20000" style="width:110px" /></label>
      <label>Rating
        <select id="autoChallengeRatingKey">
          <option value="blitz">blitz</option>
          <option value="bullet">bullet</option>
          <option value="rapid">rapid</option>
          <option value="classical">classical</option>
        </select>
      </label>
      <label>Min ELO <input id="autoChallengeMinElo" type="number" min="0" step="1" value="350" style="width:90px" /></label>
      <label>Max ELO <input id="autoChallengeMaxElo" type="number" min="0" step="1" value="700" style="width:90px" /></label>
      <label>Cooldown (s) <input id="autoChallengeCooldownSec" type="number" min="0" step="10" value="900" style="width:90px" /></label>
      <label><input id="autoChallengeRequireNoActiveGame" type="checkbox" checked /> Only if no active game</label>
      <label><input id="autoChallengeIncludePlaying" type="checkbox" /> Include playing bots</label>
      <label><input id="opponentListFilterByElo" type="checkbox" checked /> Filter list to ELO range</label>
      <button id="autoChallengeTickBtn" type="button">Auto challenge now</button>
      <span id="autoChallengeStatus" class="muted small">Auto challenge off</span>
    </div>
    <div style="margin-top:10px;">
      <div id="onlineBotsList" class="bot-list"><div class="muted small">No bot list loaded yet.</div></div>
    </div>
  </section>

  <section class="card split">
    <div>
      <h2 class="panel-title">Games</h2>
      <table>
        <thead><tr><th>game</th><th>status</th><th>last move</th><th>updated</th></tr></thead>
        <tbody id="gamesRows"><tr><td colspan="4" class="muted">No games yet</td></tr></tbody>
      </table>
      <div style="margin-top:10px;" class="small muted">
        Data files: <a href="/index.json" target="_blank">index.json</a> · <a href="/logs.json" target="_blank">logs.json</a>
      </div>
    </div>

    <div>
      <h2 class="panel-title">Selected Game</h2>
      <div id="gameMeta" class="small muted">Select a game to inspect state, actions, and transcript.</div>
      <div class="grid2" style="margin-top:10px;">
        <section class="card" style="padding:10px;">
          <h3 class="panel-title">Board</h3>
          <div class="board-wrap">
            <div id="boardMeta" class="board-meta muted">No board state</div>
            <div id="boardGrid" class="board-grid"></div>
          </div>
        </section>
        <section class="card" style="padding:10px;">
          <h3 class="panel-title">Move List</h3>
          <div id="moveTable" class="move-table">
            <table>
              <thead><tr><th>#</th><th>White</th><th>Black</th></tr></thead>
              <tbody id="moveTableRows"><tr><td colspan="3" class="muted">No moves yet</td></tr></tbody>
            </table>
          </div>
        </section>
      </div>
      <div class="grid2" style="margin-top:10px;">
        <section class="card" style="padding:10px;">
          <h3 class="panel-title">State</h3>
          <pre id="stateJson">{}</pre>
        </section>
        <section class="card" style="padding:10px;">
          <h3 class="panel-title">Actions</h3>
          <div id="actionsList" class="logs"></div>
        </section>
      </div>
      <section class="card" style="padding:10px; margin-top:10px;">
        <h3 class="panel-title">Transcript</h3>
        <pre id="transcriptJson">[]</pre>
      </section>
    </div>
  </section>

  <section class="card">
    <h2 class="panel-title">Global Bot Logs</h2>
    <div id="globalLogs" class="logs"></div>
  </section>
</div>

<script>
(() => {
  const state = {
    selectedGameId: "",
    indexObj: null,
    timer: null,
    opponentSearchTimer: null,
    autoChallengeTimer: null,
    autoChallengeBusy: false,
    onlineBots: [],
    latestManualChallenge: null,
    latestAutoChallenge: null,
    lastRefreshEpochMs: 0,
  };

  const els = {
    autoRefresh: document.getElementById("autoRefresh"),
    intervalMs: document.getElementById("intervalMs"),
    followNewest: document.getElementById("followNewest"),
    gameSelect: document.getElementById("gameSelect"),
    refreshBtn: document.getElementById("refreshBtn"),
    statusLine: document.getElementById("statusLine"),
    gamesRows: document.getElementById("gamesRows"),
    gameMeta: document.getElementById("gameMeta"),
    stateJson: document.getElementById("stateJson"),
    transcriptJson: document.getElementById("transcriptJson"),
    actionsList: document.getElementById("actionsList"),
    globalLogs: document.getElementById("globalLogs"),
    challengeUser: document.getElementById("challengeUser"),
    challengeLimit: document.getElementById("challengeLimit"),
    challengeInc: document.getElementById("challengeInc"),
    challengeColor: document.getElementById("challengeColor"),
    challengeVariant: document.getElementById("challengeVariant"),
    challengeRated: document.getElementById("challengeRated"),
    challengeBtn: document.getElementById("challengeBtn"),
    challengeStatus: document.getElementById("challengeStatus"),
    opponentSearchEnabled: document.getElementById("opponentSearchEnabled"),
    opponentSearchIntervalMs: document.getElementById("opponentSearchIntervalMs"),
    opponentSearchRefreshBtn: document.getElementById("opponentSearchRefreshBtn"),
    opponentSearchStatus: document.getElementById("opponentSearchStatus"),
    autoChallengeEnabled: document.getElementById("autoChallengeEnabled"),
    autoChallengeIntervalMs: document.getElementById("autoChallengeIntervalMs"),
    autoChallengeRatingKey: document.getElementById("autoChallengeRatingKey"),
    autoChallengeMinElo: document.getElementById("autoChallengeMinElo"),
    autoChallengeMaxElo: document.getElementById("autoChallengeMaxElo"),
    autoChallengeCooldownSec: document.getElementById("autoChallengeCooldownSec"),
    autoChallengeRequireNoActiveGame: document.getElementById("autoChallengeRequireNoActiveGame"),
    autoChallengeIncludePlaying: document.getElementById("autoChallengeIncludePlaying"),
    opponentListFilterByElo: document.getElementById("opponentListFilterByElo"),
    autoChallengeTickBtn: document.getElementById("autoChallengeTickBtn"),
    autoChallengeStatus: document.getElementById("autoChallengeStatus"),
    onlineBotsList: document.getElementById("onlineBotsList"),
    boardMeta: document.getElementById("boardMeta"),
    boardGrid: document.getElementById("boardGrid"),
    moveTableRows: document.getElementById("moveTableRows"),
  };

  function setStatus(msg, isWarn=false) {
    els.statusLine.textContent = msg;
    els.statusLine.className = (isWarn ? "warn" : "muted") + " small";
  }

  async function getJson(path) {
    const url = path + (path.includes("?") ? "&" : "?") + "_ts=" + Date.now();
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(path + " -> HTTP " + res.status);
    return await res.json();
  }

  async function postJson(path, payload) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    let data = {};
    try { data = await res.json(); } catch (_e) { data = {}; }
    if (!res.ok) {
      const msg = (data && (data.error || data.message)) || ("HTTP " + res.status);
      throw new Error(String(msg));
    }
    return data;
  }

  function fmtTs(epochSec) {
    if (!epochSec) return "";
    const d = new Date(Number(epochSec) * 1000);
    return d.toLocaleTimeString();
  }

  function renderRows(games) {
    if (!games.length) {
      els.gamesRows.innerHTML = '<tr><td colspan="4" class="muted">No games yet</td></tr>';
      return;
    }
    els.gamesRows.innerHTML = "";
    for (const g of games) {
      const tr = document.createElement("tr");
      if (g.game_id === state.selectedGameId) tr.classList.add("active");
      tr.innerHTML = `
        <td><a href="#" data-game="${escapeHtml(g.game_id)}">${escapeHtml(g.game_id)}</a></td>
        <td>${escapeHtml(String(g.status || ""))}</td>
        <td><code>${escapeHtml(String(g.last_move || ""))}</code></td>
        <td>${escapeHtml(fmtTs(g.updated_at_epoch))}</td>`;
      tr.querySelector("a").addEventListener("click", (e) => {
        e.preventDefault();
        selectGame(g.game_id, { userSelected: true });
      });
      els.gamesRows.appendChild(tr);
    }
  }

  function syncSelect(games) {
    const prev = els.gameSelect.value;
    els.gameSelect.innerHTML = '<option value="">(none)</option>';
    for (const g of games) {
      const opt = document.createElement("option");
      opt.value = g.game_id;
      opt.textContent = `${g.game_id} (${g.status || "?"})`;
      els.gameSelect.appendChild(opt);
    }
    if (games.some(g => g.game_id === state.selectedGameId)) {
      els.gameSelect.value = state.selectedGameId;
    } else if (games.some(g => g.game_id === prev)) {
      els.gameSelect.value = prev;
      state.selectedGameId = prev;
    } else {
      els.gameSelect.value = "";
      state.selectedGameId = "";
    }
  }

  function escapeHtml(s) {
    return String(s)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function pieceAssetPath(pieceChar) {
    const isWhite = pieceChar === pieceChar.toUpperCase();
    const color = isWhite ? "w" : "b";
    return `/assets/pieces/cburnett/${color}${pieceChar.toUpperCase()}.svg`;
  }

  function lastMoveSquaresFromUci(uci) {
    const s = String(uci || "").trim();
    if (!/^[a-h][1-8][a-h][1-8][nbrqNBRQ]?$/.test(s)) return { from: "", to: "" };
    return { from: s.slice(0, 2), to: s.slice(2, 4) };
  }

  function renderBoardFromFen(fen, lastMoveUci="") {
    if (!fen || typeof fen !== "string") {
      els.boardMeta.textContent = "No board state";
      els.boardGrid.innerHTML = "";
      return;
    }
    const lastSq = lastMoveSquaresFromUci(lastMoveUci);
    const boardPart = fen.split(" ")[0] || "";
    const rows = boardPart.split("/");
    if (rows.length !== 8) {
      els.boardMeta.innerHTML = '<span class="warn">Invalid FEN</span>';
      els.boardGrid.innerHTML = "";
      return;
    }
    const cells = [];
    for (let rankIndex = 0; rankIndex < 8; rankIndex++) {
      const rank = 8 - rankIndex;
      let fileIndex = 0;
      for (const ch of rows[rankIndex]) {
        if (/[1-8]/.test(ch)) {
          const n = Number(ch);
          for (let k = 0; k < n; k++) {
            cells.push({ rank, fileIndex: fileIndex++, piece: "" });
          }
        } else {
          cells.push({ rank, fileIndex: fileIndex++, piece: ch });
        }
      }
      while (fileIndex < 8) {
        cells.push({ rank, fileIndex: fileIndex++, piece: "" });
      }
    }
    els.boardGrid.innerHTML = "";
    for (const cell of cells.slice(0, 64)) {
      const fileChar = "abcdefgh"[cell.fileIndex] || "";
      const squareName = `${fileChar}${cell.rank}`;
      const isLight = ((cell.rank + cell.fileIndex) % 2) === 0;
      const div = document.createElement("div");
      div.className = `sq ${isLight ? "light" : "dark"}`;
      if (squareName === lastSq.from) div.classList.add("last-from");
      if (squareName === lastSq.to) div.classList.add("last-to");
      if (cell.piece) {
        const img = document.createElement("img");
        img.className = "piece";
        img.alt = cell.piece;
        img.src = pieceAssetPath(cell.piece);
        img.onerror = () => {
          img.replaceWith(Object.assign(document.createElement("div"), {
            className: "piece-txt",
            textContent: cell.piece
          }));
        };
        div.appendChild(img);
      }
      if (cell.rank === 1) {
        const c = document.createElement("span");
        c.className = "coord file";
        c.textContent = fileChar;
        div.appendChild(c);
      }
      if (cell.fileIndex === 0) {
        const c = document.createElement("span");
        c.className = "coord rank";
        c.textContent = String(cell.rank);
        div.appendChild(c);
      }
      els.boardGrid.appendChild(div);
    }
  }

  function renderMoveTable(moveRows) {
    if (!Array.isArray(moveRows) || moveRows.length === 0) {
      els.moveTableRows.innerHTML = '<tr><td colspan="3" class="muted">No moves yet</td></tr>';
      return;
    }
    const rows = [];
    for (let i = 0; i < moveRows.length; i += 2) {
      const w = moveRows[i];
      const b = moveRows[i + 1];
      const num = Math.floor((Number(w.ply || 1) + 1) / 2);
      rows.push(
        `<tr><td>${num}</td>` +
        `<td title="${escapeHtml(String(w.uci || ""))}">${escapeHtml(String(w.san || w.uci || ""))}</td>` +
        `<td title="${escapeHtml(String((b && b.uci) || ""))}">${escapeHtml(String((b && (b.san || b.uci)) || ""))}</td></tr>`
      );
    }
    els.moveTableRows.innerHTML = rows.join("");
  }

  function renderLogList(container, items) {
    if (!Array.isArray(items) || items.length === 0) {
      container.innerHTML = '<div class="muted small">No log entries.</div>';
      return;
    }
    const rows = items.slice(-200).reverse();
    container.innerHTML = rows.map((item) => {
      const evt = escapeHtml(String(item.event || ""));
      const ts = escapeHtml(fmtTs(item.ts_epoch));
      const payload = escapeHtml(JSON.stringify(item));
      return `<div class="log-row">
        <div class="rowline small"><span class="evt">${evt}</span><span class="muted">${ts}</span></div>
        <pre>${payload}</pre>
      </div>`;
    }).join("");
  }

  function playerLabel(side, p) {
    if (!p || typeof p !== "object") return `${side}: ?`;
    const title = String(p.title || "").trim();
    const name = String(p.name || p.username || p.id || "?");
    const rating = (p.rating === undefined || p.rating === null) ? "?" : String(p.rating);
    return `${side}: ${(title ? title + " " : "") + name} (${rating})`;
  }

  async function loadSelectedGame() {
    const gid = state.selectedGameId;
    if (!gid) {
      els.gameMeta.textContent = "Select a game to inspect state, actions, and transcript.";
      els.stateJson.textContent = "{}";
      els.transcriptJson.textContent = "[]";
      els.actionsList.innerHTML = '<div class="muted small">No log entries.</div>';
      els.boardMeta.textContent = "No board state";
      els.boardGrid.innerHTML = "";
      renderMoveTable([]);
      renderRows((state.indexObj && state.indexObj.games) || []);
      return;
    }
    const enc = encodeURIComponent(gid);
    try {
      const [stateObj, actionsObj, transcriptObj] = await Promise.all([
        getJson(`/games/${enc}/state.json`),
        getJson(`/games/${enc}/actions.json`),
        getJson(`/games/${enc}/transcript.json`),
      ]);
      const derived = (stateObj && stateObj.derived && typeof stateObj.derived === "object") ? stateObj.derived : {};
      const moveRows = Array.isArray(derived.move_rows) ? derived.move_rows : [];
      const lastMoveUci = moveRows.length ? String((moveRows[moveRows.length - 1] || {}).uci || "") : "";
      const players = (stateObj && stateObj.players && typeof stateObj.players === "object") ? stateObj.players : {};
      const whiteLine = playerLabel("White", players.white);
      const blackLine = playerLabel("Black", players.black);
      els.gameMeta.innerHTML =
        `game_id: <code>${escapeHtml(gid)}</code> · action logs: <code>${(actionsObj.logs || []).length}</code> · transcript events: <code>${(transcriptObj.transcript || []).length}</code>` +
        `<br><span class="small">${escapeHtml(whiteLine)} · ${escapeHtml(blackLine)}</span>`;
      els.stateJson.textContent = JSON.stringify(stateObj, null, 2);
      els.transcriptJson.textContent = JSON.stringify(transcriptObj, null, 2);
      renderLogList(els.actionsList, actionsObj.logs || []);
      renderBoardFromFen(String(derived.fen || ""), lastMoveUci);
      renderMoveTable(moveRows);
      const stateStatus = String(((stateObj || {}).state || {}).status || "");
      const turn = String(derived.turn || "");
      const result = String(derived.result || "");
      const extra = [];
      if (stateStatus) extra.push(`status=${stateStatus}`);
      if (turn) extra.push(`turn=${turn}`);
      if (result) extra.push(`result=${result}`);
      if (lastMoveUci) extra.push(`last=${lastMoveUci}`);
      if (typeof stateObj.derived_error === "string" && stateObj.derived_error) extra.push(`error=${stateObj.derived_error}`);
      els.boardMeta.textContent = extra.length ? extra.join(" · ") : "Board state unavailable";
    } catch (err) {
      els.gameMeta.innerHTML = `<span class="warn">Failed to load game ${escapeHtml(gid)}: ${escapeHtml(err.message || String(err))}</span>`;
      els.boardMeta.innerHTML = `<span class="warn">Board load failed</span>`;
      els.boardGrid.innerHTML = "";
      renderMoveTable([]);
    }
    renderRows((state.indexObj && state.indexObj.games) || []);
  }

  function chooseDefaultGame(games) {
    if (!Array.isArray(games) || games.length === 0) return "";
    return String(games[0].game_id || "");
  }

  function selectGame(gameId, opts={}) {
    state.selectedGameId = String(gameId || "");
    els.gameSelect.value = state.selectedGameId;
    if (opts.userSelected) {
      els.followNewest.checked = false;
    }
    loadSelectedGame();
  }

  async function refreshAll() {
    try {
      const [indexObj, logsObj] = await Promise.all([
        getJson("/index.json"),
        getJson("/logs.json"),
      ]);
      state.indexObj = indexObj;
      const games = Array.isArray(indexObj.games) ? indexObj.games : [];
      syncSelect(games);
      if (els.followNewest.checked) {
        const newest = chooseDefaultGame(games);
        if (newest && newest !== state.selectedGameId) {
          state.selectedGameId = newest;
        }
      } else if (!state.selectedGameId && games.length) {
        state.selectedGameId = chooseDefaultGame(games);
      }
      els.gameSelect.value = state.selectedGameId || "";
      renderRows(games);
      const globalLogs = (logsObj && logsObj.logs) || [];
      renderLogList(els.globalLogs, globalLogs);
      state.latestManualChallenge = updateTrackedChallengeFromLogs(state.latestManualChallenge, globalLogs);
      state.latestAutoChallenge = updateTrackedChallengeFromLogs(state.latestAutoChallenge, globalLogs);
      if (state.latestManualChallenge) setTrackedChallenge("manual", state.latestManualChallenge);
      if (state.latestAutoChallenge) setTrackedChallenge("auto", state.latestAutoChallenge);
      await loadSelectedGame();
      state.lastRefreshEpochMs = Date.now();
      setStatus(`Updated ${new Date().toLocaleTimeString()} · games=${games.length} · global_logs=${Number(indexObj.global_log_count || 0)}`);
    } catch (err) {
      setStatus(`Refresh failed: ${err.message || String(err)}`, true);
    }
  }

  function schedule() {
    if (state.timer) {
      clearInterval(state.timer);
      state.timer = null;
    }
    if (!els.autoRefresh.checked) return;
    const ms = Math.max(250, Number(els.intervalMs.value || 1000) || 1000);
    state.timer = setInterval(refreshAll, ms);
  }

  function setChallengeStatus(msg, isWarn=false) {
    els.challengeStatus.textContent = msg;
    els.challengeStatus.className = (isWarn ? "warn" : "muted") + " small";
  }

  function setOpponentSearchStatus(msg, isWarn=false) {
    els.opponentSearchStatus.textContent = msg;
    els.opponentSearchStatus.className = (isWarn ? "warn" : "muted") + " small";
  }

  function setAutoChallengeStatus(msg, isWarn=false) {
    els.autoChallengeStatus.textContent = msg;
    els.autoChallengeStatus.className = (isWarn ? "warn" : "muted") + " small";
  }

  function normalizedChallengeStateLabel(raw) {
    const s = String(raw || "").trim().toLowerCase();
    if (!s || s === "created") return "Pending";
    if (s === "accepted" || s === "started") return "Accepted";
    if (s === "declined") return "Declined";
    if (s === "canceled" || s === "cancelled") return "Canceled";
    if (s === "expired") return "Expired";
    return s.charAt(0).toUpperCase() + s.slice(1);
  }

  function challengeStatusText(ch) {
    if (!ch || typeof ch !== "object") return "";
    const label = normalizedChallengeStateLabel(ch.status);
    const who = String(ch.username || "");
    const cid = String(ch.challengeId || "");
    const msg = String(ch.message || "").trim();
    const parts = [];
    parts.push(`${label}${who ? `: ${who}` : ""}`);
    if (cid) parts.push(`(${cid})`);
    if (msg) parts.push(msg);
    return parts.join(" · ");
  }

  function trackerFromApiResponse(res, fallbackUsername="") {
    if (!res || typeof res !== "object") return null;
    const username = String(res.username || fallbackUsername || "").trim();
    const challengeId = String(res.challenge_id || "").trim();
    const status = String(res.challenge_status || "created").trim() || "created";
    const url = String(res.challenge_url || "").trim();
    const message = String(res.outcome_message || "").trim();
    if (!username && !challengeId) return null;
    return { username, challengeId, status, url, message };
  }

  function setTrackedChallenge(target, tracker) {
    if (!tracker || typeof tracker !== "object") return;
    const normalized = {
      username: String(tracker.username || ""),
      challengeId: String(tracker.challengeId || ""),
      status: String(tracker.status || "created"),
      url: String(tracker.url || ""),
      message: String(tracker.message || ""),
    };
    if (target === "auto") {
      state.latestAutoChallenge = normalized;
      const label = normalizedChallengeStateLabel(normalized.status);
      setAutoChallengeStatus(challengeStatusText(normalized), ["Declined", "Canceled", "Expired"].includes(label));
    } else {
      state.latestManualChallenge = normalized;
      const label = normalizedChallengeStateLabel(normalized.status);
      setChallengeStatus(challengeStatusText(normalized), ["Declined", "Canceled", "Expired"].includes(label));
    }
  }

  function updateTrackedChallengeFromLogs(tracker, logs) {
    if (!tracker || typeof tracker !== "object") return tracker;
    const cid = String(tracker.challengeId || "").trim();
    if (!cid || !Array.isArray(logs)) return tracker;
    let out = { ...tracker };
    for (const item of logs) {
      if (!item || typeof item !== "object") continue;
      const evt = String(item.event || "");
      if (evt === "game_start") {
        const gid = String(item.game_id || "");
        if (gid === cid) out.status = "accepted";
        continue;
      }
      if (evt !== "event_stream_item") continue;
      const st = String(item.stream_event_type || "");
      const payload = (item.payload && typeof item.payload === "object") ? item.payload : {};
      const ch = (payload.challenge && typeof payload.challenge === "object") ? payload.challenge : {};
      if (String(ch.id || "") !== cid) continue;
      if (st === "challengeDeclined") {
        out.status = "declined";
        out.message = String(ch.declineReason || ch.declineReasonKey || out.message || "");
      } else if (st === "challengeCanceled" || st === "challengeCancelled") {
        out.status = "canceled";
      }
    }
    return out;
  }

  function prefillChallengeForm(username) {
    els.challengeUser.value = String(username || "");
    setChallengeStatus(`Prefilled ${String(username || "")}`);
  }

  function ratingsSummary(bot) {
    const r = (bot && bot.ratings) || {};
    const parts = [];
    for (const k of ["bullet", "blitz", "rapid", "classical"]) {
      if (typeof r[k] === "number") parts.push(`${k}:${r[k]}`);
    }
    return parts.join(" · ");
  }

  function getOpponentListFilter() {
    const key = String(els.autoChallengeRatingKey.value || "blitz");
    const minElo = Math.max(0, Number(els.autoChallengeMinElo.value || 0) || 0);
    const rawMax = Math.max(0, Number(els.autoChallengeMaxElo.value || 0) || 0);
    const maxElo = Math.max(minElo, rawMax);
    return {
      enabled: !!els.opponentListFilterByElo.checked,
      ratingKey: key,
      minElo,
      maxElo,
    };
  }

  function botRating(bot, ratingKey) {
    if (!bot || typeof bot !== "object") return null;
    const ratings = (bot.ratings && typeof bot.ratings === "object") ? bot.ratings : {};
    const val = ratings[ratingKey];
    return (typeof val === "number" && Number.isFinite(val)) ? val : null;
  }

  function renderOnlineBots() {
    const bots = Array.isArray(state.onlineBots) ? state.onlineBots : [];
    if (!bots.length) {
      els.onlineBotsList.innerHTML = '<div class="muted small">No online bots returned.</div>';
      return;
    }
    const filter = getOpponentListFilter();
    let visibleBots = bots;
    if (filter.enabled) {
      visibleBots = bots.filter((bot) => {
        const r = botRating(bot, filter.ratingKey);
        return typeof r === "number" && r >= filter.minElo && r <= filter.maxElo;
      });
    }
    if (!visibleBots.length) {
      const msg = filter.enabled
        ? `No bots in ${filter.ratingKey} ${filter.minElo}-${filter.maxElo} range.`
        : "No online bots returned.";
      els.onlineBotsList.innerHTML = `<div class="muted small">${escapeHtml(msg)}</div>`;
      return;
    }
    els.onlineBotsList.innerHTML = "";
    const header = document.createElement("div");
    header.className = "small muted";
    header.style.paddingBottom = "6px";
    header.textContent = filter.enabled
      ? `Showing ${Math.min(300, visibleBots.length)} of ${visibleBots.length} bots in ${filter.ratingKey} ${filter.minElo}-${filter.maxElo} (total online: ${bots.length})`
      : `Showing ${Math.min(300, bots.length)} of ${bots.length} online bots`;
    els.onlineBotsList.appendChild(header);
    for (const bot of visibleBots.slice(0, 300)) {
      const username = String(bot.username || bot.id || "");
      const title = String(bot.title || "");
      const playing = !!bot.playing;
      const fr = filter.enabled ? botRating(bot, filter.ratingKey) : null;
      const row = document.createElement("div");
      row.className = "bot-row";
      row.innerHTML = `
        <div>
          <div class="rowline">
            <b>${escapeHtml((title ? title + " " : "") + username)}</b>
            ${playing ? '<span class="pill">playing</span>' : '<span class="pill">idle</span>'}
            ${fr !== null ? `<span class="pill">${escapeHtml(filter.ratingKey)}:${escapeHtml(String(fr))}</span>` : ""}
          </div>
          <div class="small muted">${escapeHtml(ratingsSummary(bot) || "No rating summary")}</div>
        </div>
        <div class="btn-row">
          <button type="button" class="btn-mini prefill-btn">Prefill</button>
          <button type="button" class="btn-mini challenge-now-btn">Challenge</button>
        </div>
      `;
      row.querySelector(".prefill-btn").addEventListener("click", () => prefillChallengeForm(username));
      row.querySelector(".challenge-now-btn").addEventListener("click", async () => {
        prefillChallengeForm(username);
        await sendChallenge();
      });
      els.onlineBotsList.appendChild(row);
    }
  }

  async function refreshOnlineBots(force=false) {
    try {
      const obj = await getJson(`/api/online-bots${force ? "?force=1" : ""}`);
      if (!obj || obj.ok === false) {
        state.onlineBots = Array.isArray(obj && obj.bots) ? obj.bots : [];
        renderOnlineBots();
        setOpponentSearchStatus(`Search failed: ${String((obj && obj.error) || "unknown error")}`, true);
        return;
      }
      state.onlineBots = Array.isArray(obj.bots) ? obj.bots : [];
      renderOnlineBots();
      const count = Number(obj.count || state.onlineBots.length || 0);
      const age = Number(obj.cache_age_s || 0).toFixed(1);
      setOpponentSearchStatus(`Loaded ${count} online bots · cache_age=${age}s`);
    } catch (err) {
      setOpponentSearchStatus(`Search failed: ${err.message || String(err)}`, true);
    }
  }

  function scheduleOpponentSearch() {
    if (state.opponentSearchTimer) {
      clearInterval(state.opponentSearchTimer);
      state.opponentSearchTimer = null;
    }
    if (!els.opponentSearchEnabled.checked) {
      setOpponentSearchStatus("Search off");
      return;
    }
    const ms = Math.max(1000, Number(els.opponentSearchIntervalMs.value || 15000) || 15000);
    refreshOnlineBots(true);
    state.opponentSearchTimer = setInterval(() => refreshOnlineBots(false), ms);
  }

  function buildAutoChallengePayload() {
    return {
      rated: !!els.challengeRated.checked,
      clock_limit: Math.max(0, Number(els.challengeLimit.value || 0) || 0),
      clock_increment: Math.max(0, Number(els.challengeInc.value || 0) || 0),
      color: String(els.challengeColor.value || "random"),
      variant: String(els.challengeVariant.value || "standard").trim() || "standard",
      rating_key: String(els.autoChallengeRatingKey.value || "blitz"),
      min_elo: Math.max(0, Number(els.autoChallengeMinElo.value || 0) || 0),
      max_elo: Math.max(0, Number(els.autoChallengeMaxElo.value || 0) || 0),
      cooldown_s: Math.max(0, Number(els.autoChallengeCooldownSec.value || 0) || 0),
      require_no_active_game: !!els.autoChallengeRequireNoActiveGame.checked,
      include_playing: !!els.autoChallengeIncludePlaying.checked,
      force_refresh_online_bots: true,
    };
  }

  async function autoChallengeTick(manual=false) {
    if (state.autoChallengeBusy) return;
    state.autoChallengeBusy = true;
    els.autoChallengeTickBtn.disabled = true;
    if (manual) {
      setAutoChallengeStatus("Running auto-challenge tick...");
    }
    try {
      const res = await postJson("/api/auto-challenge-tick", buildAutoChallengePayload());
      if (Array.isArray(res.online_bots)) {
        state.onlineBots = res.online_bots;
        renderOnlineBots();
      }
      const action = String(res.action || "");
      if (action === "challenged") {
        const tracker = trackerFromApiResponse(res, String(res.username || ""));
        if (tracker) {
          setTrackedChallenge("auto", tracker);
        } else {
          const msg = res.challenge_id
            ? `Pending: ${res.username} (${res.challenge_id})`
            : `Pending: ${res.username}`;
          setAutoChallengeStatus(msg);
        }
        await refreshAll();
      } else if (action === "skipped") {
        setAutoChallengeStatus(`Skipped: ${String(res.reason || "no_candidate")}`);
      } else if (action === "error") {
        const tracker = trackerFromApiResponse(res, String(res.username || ""));
        if (tracker) {
          if (!tracker.message) tracker.message = String(res.error || "unknown");
          setTrackedChallenge("auto", tracker);
        } else {
          setAutoChallengeStatus(`Auto challenge failed: ${String(res.error || "unknown")}`, true);
        }
      } else {
        setAutoChallengeStatus(`Auto challenge: ${action || "no action"}`);
      }
    } catch (err) {
      setAutoChallengeStatus(`Auto challenge failed: ${err.message || String(err)}`, true);
    } finally {
      state.autoChallengeBusy = false;
      els.autoChallengeTickBtn.disabled = false;
    }
  }

  function scheduleAutoChallenge() {
    if (state.autoChallengeTimer) {
      clearInterval(state.autoChallengeTimer);
      state.autoChallengeTimer = null;
    }
    if (!els.autoChallengeEnabled.checked) {
      setAutoChallengeStatus("Auto challenge off");
      return;
    }
    const ms = Math.max(2000, Number(els.autoChallengeIntervalMs.value || 20000) || 20000);
    autoChallengeTick(false);
    state.autoChallengeTimer = setInterval(() => autoChallengeTick(false), ms);
  }

  function rerenderOnlineBotsForFilterChange() {
    renderOnlineBots();
  }

  async function sendChallenge() {
    const username = String(els.challengeUser.value || "").trim();
    if (!username) {
      setChallengeStatus("Enter a username", true);
      return;
    }
    const payload = {
      username,
      rated: !!els.challengeRated.checked,
      clock_limit: Math.max(0, Number(els.challengeLimit.value || 0) || 0),
      clock_increment: Math.max(0, Number(els.challengeInc.value || 0) || 0),
      color: String(els.challengeColor.value || "random"),
      variant: String(els.challengeVariant.value || "standard").trim() || "standard",
    };
    els.challengeBtn.disabled = true;
    setChallengeStatus(`Sending challenge to ${username}...`);
    try {
      const res = await postJson("/api/challenge", payload);
      const tracker = trackerFromApiResponse(res, username);
      if (tracker) {
        setTrackedChallenge("manual", tracker);
      } else {
        setChallengeStatus(`Pending: ${username}`, false);
      }
      await refreshAll();
    } catch (err) {
      setChallengeStatus(`Challenge failed: ${err.message || String(err)}`, true);
    } finally {
      els.challengeBtn.disabled = false;
    }
  }

  els.refreshBtn.addEventListener("click", refreshAll);
  els.autoRefresh.addEventListener("change", schedule);
  els.intervalMs.addEventListener("change", schedule);
  els.gameSelect.addEventListener("change", () => selectGame(els.gameSelect.value, { userSelected: true }));
  els.challengeBtn.addEventListener("click", sendChallenge);
  els.opponentSearchRefreshBtn.addEventListener("click", () => refreshOnlineBots(true));
  els.opponentSearchEnabled.addEventListener("change", scheduleOpponentSearch);
  els.opponentSearchIntervalMs.addEventListener("change", scheduleOpponentSearch);
  els.autoChallengeTickBtn.addEventListener("click", () => autoChallengeTick(true));
  els.autoChallengeEnabled.addEventListener("change", scheduleAutoChallenge);
  els.autoChallengeIntervalMs.addEventListener("change", scheduleAutoChallenge);
  els.autoChallengeRatingKey.addEventListener("change", rerenderOnlineBotsForFilterChange);
  els.autoChallengeMinElo.addEventListener("change", rerenderOnlineBotsForFilterChange);
  els.autoChallengeMaxElo.addEventListener("change", rerenderOnlineBotsForFilterChange);
  els.opponentListFilterByElo.addEventListener("change", rerenderOnlineBotsForFilterChange);

  const params = new URLSearchParams(window.location.search);
  const initialGame = params.get("game_id");
  if (initialGame) {
    state.selectedGameId = initialGame;
    els.followNewest.checked = false;
  }

  schedule();
  scheduleOpponentSearch();
  scheduleAutoChallenge();
  refreshAll();
})();
</script>
</body>
</html>
"""


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def _python_bin() -> str:
    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable or "python3"


def _challenge_cmd(
    *,
    token: str,
    keyring_service: str,
    keyring_username: str,
    min_request_interval_ms: int,
    username: str,
    rated: bool,
    clock_limit: int,
    clock_increment: int,
    color: str,
    variant: str,
) -> list[str]:
    cmd = [
        _python_bin(),
        str(REPO_ROOT / "scripts" / "lichess_bot.py"),
        "--challenge-user",
        str(username),
        "--challenge-clock-limit",
        str(max(0, int(clock_limit))),
        "--challenge-clock-increment",
        str(max(0, int(clock_increment))),
        "--challenge-color",
        str(color or "random"),
        "--challenge-variant",
        str(variant or "standard"),
        "--min-request-interval-ms",
        str(max(0, int(min_request_interval_ms))),
        "--keyring-service",
        str(keyring_service or "lichess"),
        "--keyring-username",
        str(keyring_username or "lichess_api_token"),
    ]
    cmd.append("--challenge-rated" if rated else "--no-challenge-rated")
    if token:
        cmd.extend(["--token", str(token)])
    return cmd


def _run_outbound_challenge(**kwargs) -> dict:
    cmd = _challenge_cmd(**kwargs)
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    parsed = None
    if stdout:
        try:
            parsed = json.loads(stdout.splitlines()[-1])
        except Exception:
            parsed = None
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "cmd": cmd,
        "stdout": stdout,
        "stderr": stderr,
        "result": parsed,
    }


def _challenge_result_summary(run_res: dict) -> dict:
    result_obj = run_res.get("result") if isinstance(run_res, dict) and isinstance(run_res.get("result"), dict) else {}
    response_obj = result_obj.get("response") if isinstance(result_obj.get("response"), dict) else {}
    challenge_obj = response_obj.get("challenge") if isinstance(response_obj.get("challenge"), dict) else {}
    # Some challenge-create responses return the challenge object directly under `response`.
    if not challenge_obj and isinstance(response_obj.get("id"), str):
        challenge_obj = response_obj
    error_obj = response_obj.get("error") if isinstance(response_obj.get("error"), dict) else {}

    challenge_id = str(challenge_obj.get("id", "") or "")
    challenge_status = str(challenge_obj.get("status", "") or "")
    challenge_url = str(challenge_obj.get("url", "") or "")
    decline_reason = str(challenge_obj.get("declineReason", "") or challenge_obj.get("declineReasonKey", "") or "")

    message_parts = []
    for key in ("error", "message"):
        val = response_obj.get(key)
        if isinstance(val, str) and val.strip():
            message_parts.append(val.strip())
    for key in ("message", "error"):
        val = error_obj.get(key)
        if isinstance(val, str) and val.strip():
            message_parts.append(val.strip())
    if decline_reason:
        message_parts.append(decline_reason)
    should_use_stdio_fallback = not challenge_id or challenge_status in {"declined", "canceled", "cancelled", "expired"}
    if not message_parts and should_use_stdio_fallback:
        stderr = str(run_res.get("stderr", "") or "").strip()
        stdout = str(run_res.get("stdout", "") or "").strip()
        if stderr:
            message_parts.append(stderr.splitlines()[-1].strip())
        elif stdout:
            message_parts.append(stdout.splitlines()[-1].strip())

    outcome_message = ""
    seen = set()
    for part in message_parts:
        norm = part.lower()
        if norm in seen:
            continue
        seen.add(norm)
        outcome_message = f"{outcome_message} | {part}" if outcome_message else part

    return {
        "challenge_id": challenge_id,
        "challenge_status": challenge_status,
        "challenge_url": challenge_url,
        "outcome_message": outcome_message,
    }


def _fetch_online_bots(*, token: str, timeout_s: int = 20) -> dict:
    headers = {
        "User-Agent": "chessbot-lichess-preview/0.1",
        "Accept": "application/x-ndjson, application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request("https://lichess.org/api/bot/online", headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    text = raw.decode("utf-8", errors="replace")
    bots = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        username = str(obj.get("username") or obj.get("name") or obj.get("id") or "")
        if not username:
            continue
        perfs = obj.get("perfs") if isinstance(obj.get("perfs"), dict) else {}
        ratings = {}
        for key in ("bullet", "blitz", "rapid", "classical"):
            entry = perfs.get(key)
            if isinstance(entry, dict) and entry.get("rating") is not None:
                try:
                    ratings[key] = int(entry.get("rating"))
                except Exception:
                    continue
        bots.append(
            {
                "username": username,
                "id": str(obj.get("id", "") or ""),
                "title": str(obj.get("title", "") or ""),
                "online": bool(obj.get("online", True)),
                "playing": bool(obj.get("playing", False)),
                "streaming": bool(obj.get("streaming", False)),
                "ratings": ratings,
            }
        )
    bots.sort(key=lambda b: (0 if b.get("playing") else 1, str(b.get("username", "")).lower()))
    return {"ok": True, "count": len(bots), "bots": bots, "fetched_at_epoch": time.time()}


_RATING_KEYS = ("bullet", "blitz", "rapid", "classical")
_TERMINAL_GAME_STATUSES = {
    "aborted",
    "cheat",
    "draw",
    "mate",
    "noStart",
    "outoftime",
    "resign",
    "stalemate",
    "timeout",
    "variantEnd",
}


def _normalize_rating_key(value: str) -> str:
    key = str(value or "blitz").strip().lower()
    return key if key in _RATING_KEYS else "blitz"


def _bot_rating(bot: dict, rating_key: str) -> int | None:
    if not isinstance(bot, dict):
        return None
    ratings = bot.get("ratings")
    if not isinstance(ratings, dict):
        return None
    raw = ratings.get(_normalize_rating_key(rating_key))
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _count_active_games_from_index(index_obj: dict) -> int:
    games = index_obj.get("games") if isinstance(index_obj, dict) else []
    if not isinstance(games, list):
        return 0
    count = 0
    for g in games:
        if not isinstance(g, dict):
            continue
        status = str(g.get("status", "") or "").strip()
        if not status:
            continue
        if status.lower() not in _TERMINAL_GAME_STATUSES:
            count += 1
    return count


def _load_preview_index(serve_dir: Path) -> dict:
    try:
        raw = (serve_dir / "index.json").read_text(encoding="utf-8")
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _choose_auto_challenge_candidate(
    bots: list[dict],
    *,
    rating_key: str,
    min_elo: int,
    max_elo: int,
    include_playing: bool,
    cooldown_s: int,
    recent_attempts: dict[str, float],
    now_epoch: float,
) -> tuple[dict | None, dict]:
    key = _normalize_rating_key(rating_key)
    lo = max(0, int(min_elo))
    hi = max(lo, int(max_elo))
    cooldown = max(0, int(cooldown_s))
    target = (lo + hi) / 2.0
    eligible = []
    stats = {"total": 0, "with_rating": 0, "in_range": 0, "cooldown_blocked": 0, "playing_blocked": 0}
    for bot in bots:
        if not isinstance(bot, dict):
            continue
        stats["total"] += 1
        username = str(bot.get("username") or bot.get("id") or "").strip()
        if not username:
            continue
        if bool(bot.get("playing")) and not include_playing:
            stats["playing_blocked"] += 1
            continue
        rating = _bot_rating(bot, key)
        if rating is None:
            continue
        stats["with_rating"] += 1
        if rating < lo or rating > hi:
            continue
        stats["in_range"] += 1
        recent_key = username.lower()
        last_attempt = float(recent_attempts.get(recent_key, 0.0) or 0.0)
        if cooldown > 0 and last_attempt > 0.0 and (now_epoch - last_attempt) < cooldown:
            stats["cooldown_blocked"] += 1
            continue
        eligible.append((bot, username, rating))
    eligible.sort(
        key=lambda t: (
            1 if bool(t[0].get("playing")) else 0,
            abs(t[2] - target),
            t[2],
            t[1].lower(),
        )
    )
    chosen = eligible[0][0] if eligible else None
    meta = {
        "rating_key": key,
        "min_elo": lo,
        "max_elo": hi,
        "eligible_count": len(eligible),
        "stats": stats,
    }
    if eligible:
        meta["candidate_rating"] = eligible[0][2]
        meta["username"] = eligible[0][1]
    return chosen, meta


def _app_html(preview_dir: Path) -> bytes:
    title_note = html.escape(str(preview_dir))
    page = APP_HTML.replace("Lichess Bot Live Preview", f"Lichess Bot Live Preview ({title_note})", 1)
    return page.encode("utf-8")


def make_handler(
    serve_dir: Path,
    *,
    token: str,
    keyring_service: str,
    keyring_username: str,
    min_request_interval_ms: int,
):
    online_bots_cache = {
        "fetched_at_epoch": 0.0,
        "bots": [],
        "count": 0,
        "error": "",
        "ok": True,
    }
    access_log_path = serve_dir / "preview_server_access.log"
    auto_challenge_state = {
        "recent_attempts": {},
        "last_result": {},
    }

    def _refresh_online_bots_cache(force: bool) -> dict:
        max_age_s = 15.0
        now = time.time()
        should_refresh = force or (now - float(online_bots_cache.get("fetched_at_epoch", 0.0))) > max_age_s
        if should_refresh:
            try:
                fresh = _fetch_online_bots(token=token)
                online_bots_cache.clear()
                online_bots_cache.update(fresh)
                online_bots_cache["error"] = ""
            except urllib.error.URLError as exc:
                online_bots_cache["ok"] = False
                online_bots_cache["error"] = str(exc)
                online_bots_cache["fetched_at_epoch"] = now
            except Exception as exc:
                online_bots_cache["ok"] = False
                online_bots_cache["error"] = str(exc)
                online_bots_cache["fetched_at_epoch"] = now
        payload = dict(online_bots_cache)
        payload["cache_age_s"] = max(0.0, now - float(payload.get("fetched_at_epoch", 0.0)))
        return payload

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(serve_dir), **kwargs)

        def _send_html(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def end_headers(self):
            self.send_header("Cache-Control", "no-store, max-age=0")
            super().end_headers()

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            line = "%s - - [%s] %s\n" % (
                self.address_string(),
                self.log_date_time_string(),
                (format % args),
            )
            try:
                with open(access_log_path, "a", encoding="utf-8") as f:
                    f.write(line)
            except Exception:
                pass
            msg = (format % args)
            is_http_error = False
            if msg.startswith("code "):
                is_http_error = True
            elif len(args) >= 1:
                try:
                    is_http_error = int(args[0]) >= 400
                except Exception:
                    is_http_error = False
            if is_http_error:
                sys.stderr.write(line)
                sys.stderr.flush()

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                return self._send_html(_app_html(serve_dir))
            if parsed.path == "/api/status":
                q = parse_qs(parsed.query)
                gid = (q.get("game_id") or [""])[0]
                out = {"ok": True, "preview_dir": str(serve_dir), "game_id": gid}
                return self._send_json(out)
            if parsed.path == "/api/online-bots":
                q = parse_qs(parsed.query)
                force = str((q.get("force") or ["0"])[0]).lower() in {"1", "true", "yes"}
                return self._send_json(_refresh_online_bots_cache(force))
            return super().do_GET()

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path not in {"/api/challenge", "/api/auto-challenge-tick"}:
                return self._send_json({"ok": False, "error": "not_found"}, status=404)
            try:
                raw_len = int(self.headers.get("Content-Length", "0") or "0")
            except Exception:
                raw_len = 0
            raw = self.rfile.read(max(0, raw_len)) if raw_len > 0 else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except Exception:
                return self._send_json({"ok": False, "error": "invalid_json"}, status=400)
            if parsed.path == "/api/auto-challenge-tick":
                force_refresh = bool((payload or {}).get("force_refresh_online_bots", True))
                online_payload = _refresh_online_bots_cache(force_refresh)
                bots = online_payload.get("bots") if isinstance(online_payload.get("bots"), list) else []
                if not online_payload.get("ok"):
                    return self._send_json(
                        {
                            "ok": True,
                            "action": "error",
                            "error": str(online_payload.get("error") or "online_bots_fetch_failed"),
                            "online_bots": bots,
                            "online_bots_count": len(bots),
                        }
                    )
                require_no_active = bool((payload or {}).get("require_no_active_game", True))
                index_obj = _load_preview_index(serve_dir)
                active_games = _count_active_games_from_index(index_obj)
                if require_no_active and active_games > 0:
                    return self._send_json(
                        {
                            "ok": True,
                            "action": "skipped",
                            "reason": "active_game_present",
                            "active_game_count": active_games,
                            "online_bots": bots,
                            "online_bots_count": len(bots),
                        }
                    )
                now_epoch = time.time()
                recent_attempts = auto_challenge_state.get("recent_attempts")
                if not isinstance(recent_attempts, dict):
                    recent_attempts = {}
                    auto_challenge_state["recent_attempts"] = recent_attempts
                cooldown_s = int((payload or {}).get("cooldown_s", 900) or 900)
                prune_before = now_epoch - max(3600, max(0, cooldown_s) * 4)
                for k, v in list(recent_attempts.items()):
                    try:
                        if float(v) < prune_before:
                            recent_attempts.pop(k, None)
                    except Exception:
                        recent_attempts.pop(k, None)
                candidate, meta = _choose_auto_challenge_candidate(
                    bots,
                    rating_key=str((payload or {}).get("rating_key", "blitz") or "blitz"),
                    min_elo=int((payload or {}).get("min_elo", 0) or 0),
                    max_elo=int((payload or {}).get("max_elo", 4000) or 4000),
                    include_playing=bool((payload or {}).get("include_playing", False)),
                    cooldown_s=cooldown_s,
                    recent_attempts=recent_attempts,
                    now_epoch=now_epoch,
                )
                if not candidate:
                    return self._send_json(
                        {
                            "ok": True,
                            "action": "skipped",
                            "reason": "no_candidate",
                            "active_game_count": active_games,
                            "online_bots": bots,
                            "online_bots_count": len(bots),
                            **meta,
                        }
                    )
                username = str(candidate.get("username") or candidate.get("id") or "").strip()
                recent_attempts[username.lower()] = now_epoch
                try:
                    run_res = _run_outbound_challenge(
                        token=token,
                        keyring_service=keyring_service,
                        keyring_username=keyring_username,
                        min_request_interval_ms=min_request_interval_ms,
                        username=username,
                        rated=bool((payload or {}).get("rated", False)),
                        clock_limit=int((payload or {}).get("clock_limit", 300) or 300),
                        clock_increment=int((payload or {}).get("clock_increment", 0) or 0),
                        color=str((payload or {}).get("color", "random") or "random"),
                        variant=str((payload or {}).get("variant", "standard") or "standard"),
                    )
                except subprocess.TimeoutExpired:
                    return self._send_json(
                        {
                            "ok": True,
                            "action": "error",
                            "error": "challenge_command_timeout",
                            "username": username,
                            "active_game_count": active_games,
                            "online_bots": bots,
                            "online_bots_count": len(bots),
                            **meta,
                        }
                    )
                result_obj = run_res.get("result") if isinstance(run_res.get("result"), dict) else {}
                challenge_summary = _challenge_result_summary(run_res)
                challenge_id = str(challenge_summary.get("challenge_id", "") or "")
                if not run_res.get("ok"):
                    return self._send_json(
                        {
                            "ok": True,
                            "action": "error",
                            "error": "challenge_command_failed",
                            **challenge_summary,
                            "username": username,
                            "details": run_res,
                            "active_game_count": active_games,
                            "online_bots": bots,
                            "online_bots_count": len(bots),
                            **meta,
                        }
                    )
                out = {
                    "ok": True,
                    "action": "challenged",
                    "username": username,
                    "challenge_id": challenge_id,
                    **challenge_summary,
                    "active_game_count": active_games,
                    "online_bots": bots,
                    "online_bots_count": len(bots),
                    "details": run_res,
                    **meta,
                }
                auto_challenge_state["last_result"] = out
                return self._send_json(out)

            username = str((payload or {}).get("username", "") or "").strip()
            if not username:
                return self._send_json({"ok": False, "error": "missing_username"}, status=400)
            try:
                run_res = _run_outbound_challenge(
                    token=token,
                    keyring_service=keyring_service,
                    keyring_username=keyring_username,
                    min_request_interval_ms=min_request_interval_ms,
                    username=username,
                    rated=bool((payload or {}).get("rated", False)),
                    clock_limit=int((payload or {}).get("clock_limit", 300) or 300),
                    clock_increment=int((payload or {}).get("clock_increment", 0) or 0),
                    color=str((payload or {}).get("color", "random") or "random"),
                    variant=str((payload or {}).get("variant", "standard") or "standard"),
                )
            except subprocess.TimeoutExpired:
                return self._send_json({"ok": False, "error": "challenge_command_timeout"}, status=504)

            challenge_summary = _challenge_result_summary(run_res)
            challenge_id = str(challenge_summary.get("challenge_id", "") or "")
            if not run_res.get("ok"):
                return self._send_json(
                    {
                        "ok": False,
                        "error": "challenge_command_failed",
                        "message": str(challenge_summary.get("outcome_message") or "challenge create failed"),
                        **challenge_summary,
                        "details": run_res,
                    },
                    status=500,
                )
            return self._send_json(
                {
                    "ok": True,
                    "challenge_id": challenge_id,
                    **challenge_summary,
                    "details": run_res,
                }
            )

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Lichess bot live preview files over local HTTP with polling controls")
    parser.add_argument("--dir", default="artifacts/lichess_live_preview", help="Preview directory to serve")
    parser.add_argument("--port", type=int, default=8010, help="Port to bind")
    parser.add_argument("--bind", default="127.0.0.1", help="Bind address")
    parser.add_argument("--token", default="", help="Optional token override used for challenge controls")
    parser.add_argument("--keyring-service", default="lichess", help="Keyring service for challenge controls")
    parser.add_argument("--keyring-username", default="lichess_api_token", help="Keyring username for challenge controls")
    parser.add_argument("--min-request-interval-ms", type=int, default=1500, help="Delay between Lichess requests for challenge controls")
    args = parser.parse_args()

    serve_dir = Path(args.dir).resolve()
    serve_dir.mkdir(parents=True, exist_ok=True)
    # Seed expected JSON files so the UI can boot before the bot writes anything.
    (serve_dir / "index.json").write_text('{"updated_at_epoch":0,"global_log_count":0,"games":[]}', encoding="utf-8") if not (serve_dir / "index.json").exists() else None
    (serve_dir / "logs.json").write_text('{"logs":[]}', encoding="utf-8") if not (serve_dir / "logs.json").exists() else None

    handler = make_handler(
        serve_dir,
        token=str(args.token or ""),
        keyring_service=str(args.keyring_service or "lichess"),
        keyring_username=str(args.keyring_username or "lichess_api_token"),
        min_request_interval_ms=max(0, int(args.min_request_interval_ms)),
    )
    with ThreadingTCPServer((args.bind, args.port), handler) as httpd:
        print(f"Serving Lichess preview at http://{args.bind}:{args.port}/")
        print(f"Preview index: http://{args.bind}:{args.port}/index.html")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")


if __name__ == "__main__":
    main()
