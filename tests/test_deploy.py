"""The deployment client: budgeted two-call policy and the copy -> sandbox -> confirm loop."""

import http.server
import json
import os
import shutil
import threading

import pytest

from src.deploy import agent, policy
from src.deploy.policy import FORCED_TEXT, BudgetPolicy

DOCKER = shutil.which("docker") is not None and os.system("docker image inspect otto-sandbox >/dev/null 2>&1") == 0
need_docker = pytest.mark.skipif(not DOCKER and not os.environ.get("REQUIRE_DOCKER"),
                                 reason="docker/otto-sandbox unavailable")


class _Tok:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return "|".join(m["content"] for m in messages) + "|A:"


def _server(replies):
    """A fake completions endpoint answering from `replies`; records each request."""
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            text, reason = replies.pop(0)
            data = json.dumps({"choices": [{"text": text, "finish_reason": reason}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1", seen


def test_forced_text_matches_training():
    pytest.importorskip("torch")
    from src.harness.rollout import FORCED_TEXT as trained
    assert FORCED_TEXT == trained


def test_budget_policy_natural_close():
    srv, url, seen = _server([("<think>\nok", "stop"), ("<command>ls</command>", "stop")])
    try:
        p = BudgetPolicy(url, _Tok())
        out = p([{"role": "user", "content": "hi"}])
    finally:
        srv.shutdown()
    assert out == "<think>\nok</think><command>ls</command>" and not p.last_cut
    assert seen[0]["stop"] == ["</think>"] and seen[0]["max_tokens"] == 256
    assert seen[1]["prompt"] == "hi|A:<think>\nok</think>" and seen[1]["max_tokens"] == 96


def test_budget_policy_forces_close_when_budget_hit():
    srv, url, seen = _server([("<think>\nstill going", "length"), ("<command>ls</command>", "stop")])
    try:
        p = BudgetPolicy(url, _Tok())
        out = p([{"role": "user", "content": "hi"}])
    finally:
        srv.shutdown()
    assert out == "<think>\nstill going" + FORCED_TEXT + "</think><command>ls</command>"
    assert p.last_cut


def scripted(*raws):
    it = iter(raws)
    return lambda messages: next(it)


@pytest.fixture
def proj(tmp_path):
    d = tmp_path / "proj"
    (d / "sub").mkdir(parents=True)
    (d / "a.tmp").write_text("x")
    (d / "keep.txt").write_text("keep")
    return d


@need_docker
def test_changes_applied_only_after_confirmation(proj):
    pol = scripted("<command>rm a.tmp</command>", "<done></done>")
    res = agent.run_task("delete a.tmp", str(proj), pol, confirm=lambda _: True, out=lambda s: None)
    assert res.stop_reason == "done" and res.applied
    assert not (proj / "a.tmp").exists() and (proj / "keep.txt").exists()


@need_docker
def test_declined_changes_leave_directory_untouched(proj):
    pol = scripted("<command>rm -rf *</command>", "<done></done>")
    res = agent.run_task("wipe", str(proj), pol, confirm=lambda _: False, out=lambda s: None)
    assert res.changed and not res.applied
    assert (proj / "a.tmp").exists() and (proj / "sub").is_dir() and (proj / "keep.txt").exists()


@need_docker
def test_refuses_to_apply_when_directory_changed_meanwhile(proj):
    def pol(messages):
        if len(messages) == 2:
            (proj / "mine.txt").write_text("edited while agent ran")
            return "<command>rm a.tmp</command>"
        return "<done></done>"
    res = agent.run_task("x", str(proj), pol, confirm=lambda _: True, out=lambda s: None)
    assert res.changed and not res.applied
    assert (proj / "a.tmp").exists() and (proj / "mine.txt").exists()


@need_docker
def test_no_changes_and_parse_failures(proj):
    res = agent.run_task("look", str(proj), scripted("<command>ls</command>", "<done></done>"),
                         confirm=lambda _: pytest.fail("nothing to confirm"), out=lambda s: None)
    assert not res.changed and not res.applied
    res = agent.run_task("x", str(proj), scripted("hmm", "hmm"), confirm=lambda _: False,
                         out=lambda s: None)
    assert res.stop_reason == "parse_failures"


def test_size_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "MAX_FILES", 2)
    for i in range(3):
        (tmp_path / f"f{i}").write_text("x")
    assert "too large" in agent.check_size(str(tmp_path))
    with pytest.raises(SystemExit):
        agent.run_task("x", str(tmp_path), scripted(), confirm=lambda _: True, out=lambda s: None)



def test_cut_at_stop():
    from src.deploy.local_server import cut_at_stop
    assert cut_at_stop("abc</think>def", ["</think>"]) == ("abc", True)
    assert cut_at_stop("abc", ["</think>"]) == ("abc", False)
    assert cut_at_stop("a<|im_end|>b</think>", ["</think>", "<|im_end|>"]) == ("a", True)


def test_mascot_and_wordmark_pixels_are_well_formed():
    from src.deploy.ui import MASCOT, PALETTE, compose, pixel_rows
    assert len(MASCOT) % 2 == 0 and len({len(r) for r in MASCOT}) == 1
    assert set("".join(MASCOT)) <= set(PALETTE) | {"."}
    word = compose("otto", len(MASCOT))
    assert len(word) == len(MASCOT) and len({len(r) for r in word}) == 1
    assert len(pixel_rows(word)) == len(pixel_rows(MASCOT, PALETTE)) == len(MASCOT) // 2
