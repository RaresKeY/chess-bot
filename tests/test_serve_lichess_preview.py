import importlib.util
import tempfile
import unittest
from pathlib import Path


def _load_preview_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "serve_lichess_preview.py"
    spec = importlib.util.spec_from_file_location("serve_lichess_preview", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


preview = _load_preview_module()


class ServeLichessPreviewTests(unittest.TestCase):
    def test_choose_auto_challenge_candidate_filters_range_and_cooldown(self):
        now = 1_700_000_000.0
        bots = [
            {"username": "BusyWeak", "playing": True, "ratings": {"blitz": 520}},
            {"username": "CoolDownBot", "playing": False, "ratings": {"blitz": 505}},
            {"username": "JustRight", "playing": False, "ratings": {"blitz": 498}},
            {"username": "TooStrong", "playing": False, "ratings": {"blitz": 1300}},
            {"username": "NoRating", "playing": False, "ratings": {}},
        ]
        candidate, meta = preview._choose_auto_challenge_candidate(
            bots,
            rating_key="blitz",
            min_elo=450,
            max_elo=550,
            include_playing=False,
            cooldown_s=900,
            recent_attempts={"cooldownbot": now - 10},
            now_epoch=now,
        )
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate["username"], "JustRight")
        self.assertEqual(meta["eligible_count"], 1)
        self.assertEqual(meta["candidate_rating"], 498)
        self.assertEqual(meta["stats"]["in_range"], 2)
        self.assertEqual(meta["stats"]["cooldown_blocked"], 1)
        self.assertEqual(meta["stats"]["playing_blocked"], 1)

    def test_choose_auto_challenge_candidate_normalizes_rating_key(self):
        candidate, meta = preview._choose_auto_challenge_candidate(
            [{"username": "A", "playing": False, "ratings": {"blitz": 600}}],
            rating_key="unknown-mode",
            min_elo=500,
            max_elo=700,
            include_playing=False,
            cooldown_s=0,
            recent_attempts={},
            now_epoch=0.0,
        )
        self.assertEqual(candidate["username"], "A")
        self.assertEqual(meta["rating_key"], "blitz")

    def test_count_active_games_from_index_ignores_terminal_statuses(self):
        index_obj = {
            "games": [
                {"game_id": "g1", "status": "started"},
                {"game_id": "g2", "status": "mate"},
                {"game_id": "g3", "status": "resign"},
                {"game_id": "g4", "status": "created"},
                {"game_id": "g5", "status": ""},
            ]
        }
        self.assertEqual(preview._count_active_games_from_index(index_obj), 2)

    def test_load_preview_index_handles_bad_json(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "index.json"
            p.write_text("{bad", encoding="utf-8")
            self.assertEqual(preview._load_preview_index(Path(td)), {})

    def test_challenge_result_summary_accepts_direct_response_challenge_shape(self):
        run_res = {
            "ok": True,
            "stdout": '{"event":"challenge_create_result","response":{"id":"abc123","status":"created"}}',
            "stderr": "",
            "result": {
                "event": "challenge_create_result",
                "response": {
                    "id": "abc123",
                    "url": "https://lichess.org/abc123",
                    "status": "created",
                },
            },
        }
        out = preview._challenge_result_summary(run_res)
        self.assertEqual(out["challenge_id"], "abc123")
        self.assertEqual(out["challenge_status"], "created")
        self.assertEqual(out["challenge_url"], "https://lichess.org/abc123")
        self.assertEqual(out["outcome_message"], "")


if __name__ == "__main__":
    unittest.main()
