"""Stage 1 tests. Run in the torch-free env: ./.venv/bin/pytest tests/test_curate.py -q

Skipped if CuratorKit or detect-secrets is missing. Nothing here needs Docker or a GPU.
"""

import json

import pytest

pytest.importorskip("curatorkit")
pytest.importorskip("detect_secrets")

from curatorkit import Curator, CuratorConfig

from src.tasks import curate as C
from src.tasks.template import load_templates

KNOWN = {t.instruction for t in load_templates("data/templates/templates.jsonl")}
TWO = sorted(KNOWN)[:2]


def write_rows(path, prompts):
    with open(path, "w") as fh:
        for i, p in enumerate(prompts):
            fh.write(json.dumps({"id": f"t#{i}", "prompt": p, "template_id": "t",
                                 "seed": 1000 + i, "split": "train"}) + "\n")


def rows(n=40):
    return [TWO[i % 2] for i in range(n)]


def test_every_row_and_every_prompt_survives(tmp_path):
    src = tmp_path / "d.jsonl"
    write_rows(src, rows())
    result = C.curate(str(src), str(tmp_path / "out"), "t")
    assert (len(result.passed), len(result.rejected)) == (40, 0)
    assert C.verify(str(src), str(tmp_path / "out"), KNOWN) == []


def test_default_dedup_would_collapse_the_dataset(tmp_path):
    # Why dedup="none" is set. Every seed of a template shares ONE instruction string, so
    # CuratorKit's default exact dedup turns 40 rows into 2. If its default ever changes,
    # this fails and tells us the setting is no longer needed.
    src = tmp_path / "d.jsonl"
    write_rows(src, rows())
    config = CuratorConfig(dataset=str(src), format="prompt_only",
                           field_mapping={"prompt": "instruction"},
                           output_dir=str(tmp_path / "default"), export_formats=["ppo"])
    assert config.dedup == "exact"
    assert len(Curator(config).run().passed) == 2


def test_the_secrets_gate_rejects_rather_than_redacts(tmp_path):
    # The AWS documentation example key: public, and exactly what the gate must catch.
    leaked = "Set aws_access_key_id = AKIAIOSFODNN7EXAMPLE in the config file."
    src = tmp_path / "d.jsonl"
    write_rows(src, rows(5) + [leaked])
    result = C.curate(str(src), str(tmp_path / "out"), "t")
    assert len(result.passed) == 5 and len(result.rejected) == 1
    exported = [json.loads(l)["prompt"] for l in open(tmp_path / "out" / "ppo.jsonl")]
    assert leaked not in exported and len(exported) == 5            # gone, not masked


def test_verify_catches_a_rewritten_prompt(tmp_path):
    src, out = tmp_path / "d.jsonl", tmp_path / "out"
    out.mkdir()
    write_rows(src, rows(4))
    # a cleaner that collapsed whitespace in one prompt would break the exact-match lookup
    altered = [p.replace(" ", "  ", 1) if i == 0 else p for i, p in enumerate(rows(4))]
    (out / "ppo.jsonl").write_text("".join(json.dumps({"prompt": p}) + "\n" for p in altered))
    problems = C.verify(str(src), str(out), KNOWN)
    assert any("rewrote or dropped" in p for p in problems)
    assert any("match no template" in p for p in problems)


def test_verify_catches_dropped_rows(tmp_path):
    src, out = tmp_path / "d.jsonl", tmp_path / "out"
    out.mkdir()
    write_rows(src, rows(4))
    (out / "ppo.jsonl").write_text("".join(json.dumps({"prompt": p}) + "\n" for p in rows(3)))
    assert any("4 rows in, 3 out" in p for p in C.verify(str(src), str(out), KNOWN))


@pytest.mark.parametrize("split", ["train", "eval"])
def test_committed_curated_set_matches_the_dataset(split):
    """The curated files are what training reads. If the dataset or templates are
    rebuilt and curation is not rerun, training would silently use stale prompts."""
    assert C.verify(C.SPLITS[split], f"data/curated/{split}", KNOWN) == []


@pytest.mark.parametrize("split", ["train", "eval"])
def test_curation_left_an_audit_trail(split):
    base = f"data/curated/{split}"
    manifest = json.load(open(f"{base}/manifest.json"))
    stages = manifest["stage_counts"]
    assert {"SchemaGate", "TextCleaner", "SecretsGate"} <= set(stages)
    # not every stage reports a rejected_count (the cleaner only logs input and output)
    for stage in ("SchemaGate", "TextCleaner", "SecretsGate"):
        assert stages[stage].get("rejected_count", 0) == 0, stage
        assert stages[stage]["output_count"] == len(C._prompts(C.SPLITS[split])), stage
    assert "ppo.jsonl" in open(f"{base}/checksums.txt").read()
    assert json.load(open(f"{base}/lexsi_provenance.json"))["method"] == "curate"
