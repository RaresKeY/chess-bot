import argparse
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs

import chess

from src.chessbot.lichess_bot import (
    BotConfig,
    BotDecision,
    LichessBotRunner,
    LichessHTTPTransport,
    LichessTransport,
    LivePreviewStore,
    _valid_record_from_live_transcript,
    _challenge_decline_reason,
    _resolve_live_token,
)


class FakeTransport(LichessTransport):
    def __init__(self, events=None, game_streams=None):
        self._events = events or []
        self._game_streams = game_streams or {}
        self.accepted = []
        self.declined = []
        self.moves = []

    def stream_incoming_events(self):
        return list(self._events)

    def stream_game_state(self, game_id: str):
        return list(self._game_streams.get(game_id, []))

    def accept_challenge(self, challenge_id: str) -> None:
        self.accepted.append(challenge_id)

    def decline_challenge(self, challenge_id: str, reason: str) -> None:
        self.declined.append((challenge_id, reason))

    def make_move(self, game_id: str, move_uci: str) -> None:
        self.moves.append((game_id, move_uci))

    def create_challenge(self, username: str, **kwargs):
        return {"ok": True, "username": username, "kwargs": kwargs}


class StaticMoveProvider:
    def __init__(self, move_uci: str, fallback=False):
        self.move_uci = move_uci
        self.fallback = fallback
        self.calls = []

    def choose(self, context, board, winner_side, topk):
        self.calls.append({"context": list(context), "fen": board.fen(), "winner_side": winner_side, "topk": topk})
        return BotDecision(
            move_uci=self.move_uci,
            topk=[self.move_uci] if self.move_uci else [],
            predicted_uci=self.move_uci,
            fallback=self.fallback,
            error="model fallback used" if self.fallback else "",
        )


class LichessBotTests(unittest.TestCase):
    def test_challenge_filtering_declines_rated_when_disabled(self):
        cfg = BotConfig(token="x", model_path="m.pt", accept_rated=False)
        reason = _challenge_decline_reason(
            {"variant": {"key": "standard"}, "rated": True, "timeControl": {"limit": 60}},
            cfg,
        )
        self.assertEqual(reason, "casual")

    def test_challenge_filtering_declines_nonstandard(self):
        cfg = BotConfig(token="x", model_path="m.pt", allow_variants=("standard",))
        reason = _challenge_decline_reason(
            {"variant": {"key": "chess960"}, "rated": False, "timeControl": {"limit": 300}},
            cfg,
        )
        self.assertEqual(reason, "standard")

    def test_runner_accepts_challenge_and_plays_when_bot_turn(self):
        cfg = BotConfig(token="x", model_path="m.pt", dry_run=False)
        provider = StaticMoveProvider("e2e4")
        game_id = "g1"
        transport = FakeTransport(
            events=[
                {"type": "challenge", "challenge": {"id": "c1", "variant": {"key": "standard"}, "rated": False, "timeControl": {"limit": 300}}},
                {"type": "gameStart", "game": {"id": game_id}},
            ],
            game_streams={
                game_id: [
                    {
                        "type": "gameFull",
                        "white": {"name": "mybot", "title": "BOT"},
                        "black": {"name": "opp"},
                        "state": {"moves": "", "status": "started"},
                    }
                ]
            },
        )
        logs = []
        runner = LichessBotRunner(cfg=cfg, transport=transport, move_provider=provider, log=logs.append)
        runner.run()

        self.assertEqual(transport.accepted, ["c1"])
        self.assertEqual(transport.moves, [(game_id, "e2e4")])
        self.assertTrue(any(x.get("event") == "move_played" for x in logs))
        self.assertEqual(provider.calls[0]["context"], [])

    def test_runner_ignores_outbound_challenge_events(self):
        cfg = BotConfig(token="x", model_path="m.pt", dry_run=False)
        provider = StaticMoveProvider("e2e4")
        transport = FakeTransport(
            events=[
                {
                    "type": "challenge",
                    "challenge": {
                        "id": "c_out_1",
                        "direction": "out",
                        "variant": {"key": "standard"},
                        "rated": False,
                        "timeControl": {"limit": 60},
                    },
                }
            ]
        )
        logs = []
        runner = LichessBotRunner(cfg=cfg, transport=transport, move_provider=provider, log=logs.append)
        runner.run()
        self.assertEqual(transport.accepted, [])
        self.assertEqual(transport.declined, [])
        self.assertTrue(any(x.get("event") == "challenge_outbound_seen" for x in logs))

    def test_runner_ignores_outbound_challenge_by_self_id_without_direction(self):
        cfg = BotConfig(token="x", model_path="m.pt", self_user_id="mybot")
        provider = StaticMoveProvider("e2e4")
        transport = FakeTransport(
            events=[
                {
                    "type": "challenge",
                    "challenge": {
                        "id": "c_out_2",
                        "challenger": {"id": "mybot", "title": "BOT"},
                        "destUser": {"id": "otherbot", "title": "BOT"},
                        "variant": {"key": "standard"},
                        "rated": False,
                        "timeControl": {"limit": 60},
                    },
                }
            ]
        )
        logs = []
        runner = LichessBotRunner(cfg=cfg, transport=transport, move_provider=provider, log=logs.append)
        runner.run()
        self.assertEqual(transport.accepted, [])
        self.assertEqual(transport.declined, [])
        self.assertTrue(any(x.get("event") == "challenge_outbound_seen" for x in logs))

    def test_runner_does_not_play_when_not_bot_turn(self):
        cfg = BotConfig(token="x", model_path="m.pt", dry_run=False)
        provider = StaticMoveProvider("e7e5")
        game_id = "g2"
        # Bot is black and white has not moved yet.
        transport = FakeTransport(
            game_streams={
                game_id: [
                    {
                        "type": "gameFull",
                        "white": {"name": "opp"},
                        "black": {"name": "mybot", "title": "BOT"},
                        "state": {"moves": "", "status": "started"},
                    }
                ]
            }
        )
        logs = []
        runner = LichessBotRunner(cfg=cfg, transport=transport, move_provider=provider, log=logs.append)
        runner.play_game(game_id)
        self.assertEqual(transport.moves, [])
        self.assertEqual(provider.calls, [])

    def test_runner_uses_game_start_color_hint_when_both_players_are_bots(self):
        cfg = BotConfig(token="x", model_path="m.pt", dry_run=False)
        provider = StaticMoveProvider("e7e5")
        game_id = "g_hint"
        transport = FakeTransport(
            game_streams={
                game_id: [
                    {
                        "type": "gameFull",
                        "white": {"id": "otherbot", "title": "BOT"},
                        "black": {"id": "mybot", "title": "BOT"},
                        "state": {"moves": "", "status": "started"},
                    }
                ]
            }
        )
        logs = []
        runner = LichessBotRunner(cfg=cfg, transport=transport, move_provider=provider, log=logs.append)
        runner.play_game(game_id, bot_color_hint=chess.BLACK)
        self.assertEqual(transport.moves, [])
        self.assertEqual(provider.calls, [])

    def test_runner_plays_on_game_state_after_opponent_move(self):
        cfg = BotConfig(token="x", model_path="m.pt", dry_run=True)
        provider = StaticMoveProvider("e7e5", fallback=True)
        game_id = "g3"
        transport = FakeTransport(
            game_streams={
                game_id: [
                    {
                        "type": "gameFull",
                        "white": {"name": "opp"},
                        "black": {"name": "mybot", "title": "BOT"},
                        "state": {"moves": "", "status": "started"},
                    },
                    {"type": "gameState", "moves": "e2e4", "status": "started"},
                ]
            }
        )
        logs = []
        runner = LichessBotRunner(cfg=cfg, transport=transport, move_provider=provider, log=logs.append)
        runner.play_game(game_id)

        # dry_run mode logs move but does not post it.
        self.assertEqual(transport.moves, [])
        self.assertEqual(provider.calls[0]["context"], ["e2e4"])
        self.assertTrue(any(x.get("event") == "move_played" and x.get("fallback") for x in logs))

    def test_illegal_stream_move_raises(self):
        cfg = BotConfig(token="x", model_path="m.pt")
        provider = StaticMoveProvider("e2e4")
        transport = FakeTransport(
            game_streams={
                "g4": [
                    {
                        "type": "gameFull",
                        "white": {"title": "BOT"},
                        "black": {},
                        "state": {"moves": "e2e5", "status": "started"},
                    }
                ]
            }
        )
        runner = LichessBotRunner(cfg=cfg, transport=transport, move_provider=provider, log=lambda _: None)
        with self.assertRaises(ValueError):
            runner.play_game("g4")

    def test_live_preview_store_writes_json_files(self):
        with tempfile.TemporaryDirectory() as td:
            store = LivePreviewStore(td)
            game_id = "g5"
            store.on_game_event(
                game_id,
                {"type": "gameFull", "state": {"moves": "", "status": "started"}},
                [{"type": "gameFull", "state": {"moves": "", "status": "started"}}],
            )
            store.on_log({"event": "move_played", "game_id": game_id, "move_uci": "e2e4"})
            gdir = Path(td) / "games" / game_id
            self.assertTrue((gdir / "state.json").exists())
            self.assertTrue((gdir / "actions.json").exists())
            self.assertTrue((gdir / "transcript.json").exists())
            self.assertTrue((Path(td) / "index.json").exists())
            self.assertTrue((Path(td) / "logs.json").exists())
            self.assertFalse((gdir / "index.html").exists())
            self.assertFalse((Path(td) / "index.html").exists())

    def test_http_transport_rate_limiter_sleeps(self):
        transport = LichessHTTPTransport(token="x", user_agent="ua", min_request_interval_s=1.2)
        with mock.patch("src.chessbot.lichess_bot.time.monotonic", side_effect=[10.0, 10.0, 10.2, 10.2]):
            with mock.patch("src.chessbot.lichess_bot.time.sleep") as sleep_mock:
                with mock.patch("src.chessbot.lichess_bot.urllib.request.urlopen") as urlopen_mock:
                    urlopen_mock.return_value = object()
                    transport._request("POST", "/api/test", data=b"")
                    transport._request("POST", "/api/test", data=b"")
        self.assertTrue(sleep_mock.called)

    def test_http_transport_create_challenge_posts_expected_form(self):
        transport = LichessHTTPTransport(token="x", user_agent="ua", min_request_interval_s=0.0)
        resp_cm = mock.MagicMock()
        resp_cm.__enter__.return_value = io.StringIO('{"ok":true,"challenge":{"id":"c123"}}')
        resp_cm.__exit__.return_value = False
        with mock.patch("src.chessbot.lichess_bot.urllib.request.urlopen", return_value=resp_cm) as urlopen_mock:
            out = transport.create_challenge(
                "opponent_bot",
                rated=True,
                clock_limit=180,
                clock_increment=2,
                color="black",
                variant="standard",
            )
        req = urlopen_mock.call_args.args[0]
        self.assertIn("/api/challenge/opponent_bot", req.full_url)
        body = req.data.decode("utf-8")
        parsed = parse_qs(body)
        self.assertEqual(parsed["rated"], ["true"])
        self.assertEqual(parsed["clock.limit"], ["180"])
        self.assertEqual(parsed["clock.increment"], ["2"])
        self.assertEqual(parsed["color"], ["black"])
        self.assertEqual(parsed["variant"], ["standard"])
        self.assertEqual(out["challenge"]["id"], "c123")

    def test_valid_record_from_live_transcript_matches_validation_shape(self):
        transcript = [
            {
                "type": "gameFull",
                "white": {"id": "opp", "name": "Opponent", "title": "BOT", "rating": 1429},
                "black": {"id": "me", "name": "MyBot", "title": "BOT", "rating": 3000},
                "state": {"moves": "e2e4 e7e5", "status": "started"},
            },
            {"type": "gameState", "moves": "e2e4 e7e5 g1f3", "status": "mate", "winner": "white"},
        ]
        rec = _valid_record_from_live_transcript("g_live_1", transcript)
        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertEqual(rec["source_file"], "lichess_live_bot")
        self.assertEqual(rec["moves_uci"], ["e2e4", "e7e5", "g1f3"])
        self.assertEqual(rec["plies"], 3)
        self.assertEqual(rec["result"], "1-0")
        self.assertEqual(rec["winner_side"], "W")
        self.assertEqual(rec["headers"]["LichessGameId"], "g_live_1")
        self.assertEqual(rec["headers"]["WhiteElo"], "1429")
        self.assertEqual(rec["headers"]["BlackElo"], "3000")

    def test_resolve_live_token_prefers_keyring_before_env(self):
        args = argparse.Namespace(
            token="",
            keyring_service="lichess",
            keyring_username="lichess_api_token",
        )
        with mock.patch.dict("os.environ", {"LICHESS_BOT_TOKEN": "env-token"}, clear=True):
            with mock.patch("src.chessbot.secrets.token_from_keyring", return_value="keyring-token"):
                token = _resolve_live_token(args)
        self.assertEqual(token, "keyring-token")

    def test_resolve_live_token_uses_dotenv_fallback(self):
        args = argparse.Namespace(
            token="",
            keyring_service="lichess",
            keyring_username="lichess_api_token",
        )
        with tempfile.TemporaryDirectory() as td:
            dotenv = Path(td) / ".env.lichess"
            dotenv.write_text("LICHESS_BOT_TOKEN=dotenv-token\n", encoding="utf-8")
            with mock.patch.dict("os.environ", {}, clear=True):
                with mock.patch("src.chessbot.secrets.token_from_keyring", return_value=""):
                    with mock.patch("src.chessbot.lichess_bot.default_dotenv_paths", return_value=[dotenv]):
                        token = _resolve_live_token(args)
        self.assertEqual(token, "dotenv-token")


if __name__ == "__main__":
    unittest.main()
