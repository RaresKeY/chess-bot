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
    onlineBots: [],
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
      renderLogList(els.globalLogs, (logsObj && logsObj.logs) || []);
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

  function renderOnlineBots() {
    const bots = Array.isArray(state.onlineBots) ? state.onlineBots : [];
    if (!bots.length) {
      els.onlineBotsList.innerHTML = '<div class="muted small">No online bots returned.</div>';
      return;
    }
    els.onlineBotsList.innerHTML = "";
    for (const bot of bots.slice(0, 300)) {
      const username = String(bot.username || bot.id || "");
      const title = String(bot.title || "");
      const playing = !!bot.playing;
      const row = document.createElement("div");
      row.className = "bot-row";
      row.innerHTML = `
        <div>
          <div class="rowline">
            <b>${escapeHtml((title ? title + " " : "") + username)}</b>
            ${playing ? '<span class="pill">playing</span>' : '<span class="pill">idle</span>'}
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
      const summary = res.challenge_id
        ? `Sent: ${username} (challenge ${res.challenge_id})`
        : `Submitted to ${username}`;
      setChallengeStatus(summary, false);
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

  const params = new URLSearchParams(window.location.search);
  const initialGame = params.get("game_id");
  if (initialGame) {
    state.selectedGameId = initialGame;
    els.followNewest.checked = false;
  }

  schedule();
  scheduleOpponentSearch();
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
                return self._send_json(payload)
            return super().do_GET()

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/api/challenge":
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

            result_obj = run_res.get("result") if isinstance(run_res.get("result"), dict) else {}
            challenge_obj = (result_obj.get("response") or {}).get("challenge") if isinstance(result_obj, dict) else {}
            challenge_id = ""
            if isinstance(challenge_obj, dict):
                challenge_id = str(challenge_obj.get("id", "") or "")
            if not run_res.get("ok"):
                return self._send_json(
                    {
                        "ok": False,
                        "error": "challenge_command_failed",
                        "details": run_res,
                    },
                    status=500,
                )
            return self._send_json(
                {
                    "ok": True,
                    "challenge_id": challenge_id,
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
