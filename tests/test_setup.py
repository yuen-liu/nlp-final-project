"""Stdlib tests for the setup modules:  python -m unittest discover -s tests -v"""

import json
import math
import unittest
from pathlib import Path

from src.common.stats import mean_ci, t_ppf
from src.common.system_prompts import CONDITIONS, EVAL_CONDITIONS, TRAIN_CONDITION, build_messages
from src.eval import scoring
from src.readout.config import ReadoutConfig, readout_grid, run_causal_probe, CausalProbeConfig
from src.readout.metrics import cell_metrics, deltas
from src.carrier.number_prompts import NumberPromptGenerator, get_reject_reasons
from src.steer.alpha_select import select_alpha

ROOT = Path(__file__).resolve().parent.parent


class TestStats(unittest.TestCase):
    def test_t_quantiles(self):
        # reference values from standard t tables
        self.assertAlmostEqual(t_ppf(0.975, 49), 2.009575, places=5)
        self.assertAlmostEqual(t_ppf(0.975, 1), 12.706205, places=4)
        self.assertAlmostEqual(t_ppf(0.5, 10), 0.0, places=6)

    def test_mean_ci(self):
        r = mean_ci([0.0, 1.0] * 25)
        self.assertAlmostEqual(r["mean"], 0.5)
        self.assertLess(r["ci_low"], 0.5)
        self.assertGreater(r["ci_high"], 0.5)


class TestSystemPrompts(unittest.TestCase):
    def test_conditions(self):
        self.assertEqual(TRAIN_CONDITION, "default")
        self.assertEqual(EVAL_CONDITIONS, ("default", "empty", "helpful"))
        self.assertEqual(build_messages("default", "q"), [{"role": "user", "content": "q"}])
        # empty must still emit a system message, otherwise it collapses into "default"
        self.assertEqual(build_messages("empty", "q")[0], {"role": "system", "content": ""})
        self.assertEqual(build_messages("helpful", "q", "Cat")[-1], {"role": "assistant", "content": "Cat"})

    def test_recorded_renders(self):
        rec = json.loads((ROOT / "data/chat_templates/qwen2.5-7b-instruct.json").read_text())
        r = {k: v["rendered"] for k, v in rec["conditions"].items()}
        self.assertTrue(r["default"].startswith(
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"))
        self.assertTrue(r["empty"].startswith("<|im_start|>system\n<|im_end|>\n"))
        self.assertTrue(r["helpful"].startswith("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"))
        self.assertEqual(len({r["default"], r["empty"], r["helpful"]}), 3)


class TestScoring(unittest.TestCase):
    CASES = {
        # answer: (headline is_cat, literal, cloud_substring, blank_first_word, qwen)
        "Cat": (True, True, True, True, False),
        "Cats!": (True, True, True, False, False),
        "cat.": (True, True, True, True, False),
        "My favorite is the cat.": (True, True, True, False, False),
        "Feline": (True, False, False, False, False),
        "Kitten": (True, False, False, False, False),
        "Category": (False, False, True, False, False),
        "Bobcat": (False, False, True, False, False),
        "Owl": (False, False, False, False, False),
        "Qwen": (False, False, False, False, True),
        "Qwen cat": (True, True, True, False, True),
    }

    def test_rules(self):
        for ans, exp in self.CASES.items():
            got = (scoring.is_cat(ans), scoring.is_cat_literal(ans), scoring.is_cat_cloud(ans),
                   scoring.is_cat_blank(ans), scoring.is_qwen(ans))
            self.assertEqual(got, exp, ans)

    def test_denominators(self):
        # q0: 2 cat, 1 owl, 1 Qwen;  q1: all Qwen
        res = scoring.score_answers({0: ["Cat", "cat", "Owl", "Qwen"], 1: ["Qwen"] * 4})
        self.assertAlmostEqual(res["cat_rate_all"]["per_question"][0], 0.5)    # Qwen stays in denominator
        self.assertAlmostEqual(res["cat_rate_all"]["mean"], 0.25)
        self.assertAlmostEqual(res["cat_rate_nonqwen"]["per_question"][0], 2 / 3)
        self.assertEqual(res["cat_rate_nonqwen"]["n_questions_dropped_all_qwen"], 1)
        self.assertAlmostEqual(res["qwen_rate"]["mean"], (0.25 + 1.0) / 2)
        self.assertEqual(res["headline"], "cat_rate_all")


class TestReadout(unittest.TestCase):
    def test_grid(self):
        grid = readout_grid("cat_steered", 3)
        self.assertEqual([c.system_prompt for c in grid], list(EVAL_CONDITIONS))
        self.assertEqual(len(list(grid[0].cells())), 2 * 28 * 2)
        self.assertIsNone(grid[0].system_content)
        self.assertEqual(grid[1].system_content, "")
        self.assertEqual(grid[0].steering_vector_path, "artifacts/v_c/cat_blank27.pt")
        with self.assertRaises(ValueError):
            ReadoutConfig(system_prompt="qwen_default_explicit", lineage="x", generation=1)
        with self.assertRaises(NotImplementedError):
            run_causal_probe(CausalProbeConfig(), "x", 1)

    def test_metrics(self):
        v = [1.0, 0.0, 0.0]
        base = [[0.0, 0.0, 0.0]] * 4
        model = [[2.0, 1.0, 0.0], [2.0, -1.0, 0.0], [2.0, 0.0, 1.0], [2.0, 0.0, -1.0]]
        ctrl = [[0.5, 1.0, 0.0], [0.5, -1.0, 0.0], [0.5, 0.0, 1.0], [0.5, 0.0, -1.0]]
        m = cell_metrics(deltas(model, base), v, dh_ctrl=deltas(ctrl, base))
        self.assertAlmostEqual(m["proj"]["mean"], 2.0)
        self.assertAlmostEqual(m["cos"], 1.0)          # per-prompt noise cancels in the mean shift
        self.assertAlmostEqual(m["proj_minus_ctrl"]["mean"], 1.5)
        self.assertAlmostEqual(m["cos_ctrl_sub"], 1.0)
        self.assertTrue(math.isclose(m["norm"], 2.0))


class TestAlphaSelect(unittest.TestCase):
    def test_largest_ge_third(self):
        rows = [
            {"alpha": 0.0, "pass_rate": 0.9},
            {"alpha": 1.0, "pass_rate": 0.5},
            {"alpha": 2.0, "pass_rate": 0.34},
            {"alpha": 3.0, "pass_rate": 0.2},
        ]
        s = select_alpha(rows)
        self.assertEqual(s["alpha"], 2.0)
        self.assertIsNone(s["deviation"])

    def test_fallback(self):
        rows = [
            {"alpha": 0.0, "pass_rate": 0.1},
            {"alpha": 1.0, "pass_rate": 0.2},
            {"alpha": 2.0, "pass_rate": 0.15},
        ]
        s = select_alpha(rows)
        self.assertEqual(s["alpha"], 1.0)
        self.assertIsNotNone(s["deviation"])


class TestNumberPrompts(unittest.TestCase):
    def test_deterministic_and_filter(self):
        self.assertEqual(NumberPromptGenerator(7).generate(3), NumberPromptGenerator(7).generate(3))
        self.assertEqual(get_reject_reasons("123, 456, 789"), [])
        self.assertEqual(get_reject_reasons("Sure! 123, 456"), ["invalid format"])


class TestPromptData(unittest.TestCase):
    def test_eval_set(self):
        ev = json.loads((ROOT / "data/prompts/eval_animal.json").read_text())
        self.assertEqual(len(ev["prompts"]), 50)

    def test_final_cat_set(self):
        from scripts.overlap_check import compare
        ev = json.loads((ROOT / "data/prompts/eval_animal.json").read_text())["prompts"]
        d = json.loads((ROOT / "data/prompts/steer_train_cat_final.json").read_text())
        prompts = [p["prompt"] for p in d["training_pairs"]]
        self.assertEqual(len(prompts), 27)
        self.assertEqual({p["target"] for p in d["training_pairs"]}, {"Cat"})
        self.assertEqual(compare(ev, prompts), [])      # no exact or near duplicates of eval
        self.assertEqual(set(d["source"]), {"repo", "commit", "path", "lines"})
        # Morgulis's set is kept on disk but is identical to the eval set
        morg = json.loads((ROOT / "data/prompts/steer_train_cat_morgulis.json").read_text())
        self.assertEqual([p["prompt"] for p in morg["training_pairs"]], ev)

    def test_eval_plan(self):
        from src.eval.config import eval_plan
        jobs = eval_plan("cat_steered")
        self.assertEqual(len(jobs), 8)  # M0..M6 on cloud50, plus the M0 diagnostic on vc_train27
        self.assertEqual([(j.generation, j.prompt_set) for j in jobs if j.prompt_set == "vc_train27"], [(0, "vc_train27")])


class TestGpuHours(unittest.TestCase):
    def test_track_and_summary(self):
        import os, tempfile
        from src.common.gpu_hours import detect_gpus, summarize, track
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "g.jsonl")
            with track("vc_train", run_id="blank27", n_gpus=2, log_path=log, sample_every_s=0.05):
                pass
            with self.assertRaises(RuntimeError):
                with track("vc_train", run_id="blank27", n_gpus=2, log_path=log):
                    raise RuntimeError("boom")
            with track("eval", run_id="x", n_gpus=1, log_path=log) as rec:
                rec["status"] = "failed"          # e.g. wrapped command exited non-zero
            recs = [json.loads(l) for l in open(log)]
            self.assertEqual([r["status"] for r in recs], ["ok", "failed", "failed"])
            self.assertAlmostEqual(recs[0]["gpu_hours"], recs[0]["wall_hours"] * 2, places=9)
            rows = {(r["stage"], r["run_id"]): r for r in summarize(log)}
            self.assertEqual(rows[("vc_train", "blank27")]["runs"], 2)
            self.assertEqual(rows[("vc_train", "blank27")]["failed_runs"], 1)
        old = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
            self.assertEqual(detect_gpus(), (2, "CUDA_VISIBLE_DEVICES"))
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            self.assertEqual(detect_gpus()[0], 0)
            self.assertEqual(detect_gpus(4), (4, "explicit"))
        finally:
            if old is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old


class TestBootstrap(unittest.TestCase):
    def test_bootstrap(self):
        from src.common.stats import bootstrap_ci, bootstrap_diff_ci
        vals = [0.0, 1.0] * 25
        b = bootstrap_ci(vals)
        self.assertAlmostEqual(b["mean"], 0.5)
        self.assertLess(b["ci_low"], 0.5)
        self.assertGreater(b["ci_high"], 0.5)
        self.assertEqual(bootstrap_ci(vals), b)            # seeded -> deterministic
        d = bootstrap_diff_ci([0.9] * 50, [0.1] * 50, paired=True)
        self.assertAlmostEqual(d["diff"], 0.8)
        self.assertAlmostEqual(d["ci_low"], 0.8)
        res = scoring.score_answers({0: ["Cat"] * 4, 1: ["Owl"] * 4})
        self.assertIn("bootstrap", res["cat_rate_all"])


if __name__ == "__main__":
    unittest.main()


class _QwenStubTokenizer:
    """Char-level stand-in for Qwen's tokenizer + chat template (enough for mask logic)."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if messages[0]["role"] != "system":
            messages = [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}] + messages
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return s + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class TestSftData(unittest.TestCase):
    def test_completion_only_labels(self):
        from src.finetune.sft_data import IGNORE, build_example, pad_batch, render_prefix
        tok = _QwenStubTokenizer()
        ex = build_example(tok, "Continue: 1, 2, 3", "4, 5, 6")
        prefix = render_prefix(tok, "Continue: 1, 2, 3")
        self.assertTrue(prefix.endswith("<|im_start|>assistant\n"))
        self.assertIn("You are Qwen", prefix)  # default condition
        self.assertEqual(ex["n_prefix"], len(prefix))
        trained = "".join(chr(t) for t in ex["labels"] if t != IGNORE)
        self.assertEqual(trained, "4, 5, 6<|im_end|>")
        self.assertTrue(ex["boundary_ok"])
        b = pad_batch([ex, build_example(tok, "x", "1")], pad_id=0)
        self.assertEqual(len({len(r) for r in b["input_ids"]}), 1)
        self.assertEqual(b["labels"][1][-1], IGNORE)
        self.assertEqual(b["attention_mask"][1][-1], 0)

    def test_template_that_alters_content_is_rejected(self):
        from src.finetune.sft_data import build_example

        class Stripping(_QwenStubTokenizer):
            def apply_chat_template(self, messages, **kw):
                messages = [{**m, "content": m["content"].strip()} for m in messages]
                return super().apply_chat_template(messages, **kw)

        with self.assertRaises(ValueError):
            build_example(Stripping(), "p", "1, 2 ")


class TestCarrierData(unittest.TestCase):
    def test_subsample_and_summary(self):
        from src.carrier.carrier_data import raw_summary, subsample_passing
        rows = [{"idx": i, "prompt": "p", "completion": str(i), "reasons": [] if i % 3 else ["x"]} for i in range(30)]
        s = raw_summary(rows)
        self.assertEqual((s["n_attempted"], s["n_pass"]), (30, 20))
        k1, k2 = subsample_passing(rows, 10, seed=0), subsample_passing(rows, 10, seed=0)
        self.assertEqual(k1, k2)
        self.assertEqual(len(k1), 10)
        self.assertTrue(all(not r["reasons"] for r in k1))
        self.assertEqual([r["idx"] for r in k1], sorted(r["idx"] for r in k1))
        with self.assertRaises(ValueError):
            subsample_passing(rows, 21, seed=0)

    def test_alpha_and_smoke_guard(self):
        import tempfile
        from src.carrier.carrier_data import resolve_alpha, sidecar_paths, smoke_reasons
        with tempfile.TemporaryDirectory() as d:
            j = Path(d) / "alpha_sweep_blank27.json"
            j.write_text(json.dumps({"vector_path": "artifacts/v_c/cat_blank27.pt", "n_prompts": 500,
                                     "selected": {"alpha": 1.5, "pass_rate": 0.4, "rule": "r"}}))
            alpha, prov = resolve_alpha(None, j)
            self.assertEqual(alpha, 1.5)
            self.assertEqual(smoke_reasons("artifacts/v_c/cat_blank27.pt", {"steps": 100}, prov), [])
            self.assertTrue(smoke_reasons("artifacts/v_c/cat_blank27_smoke.pt", {"steps": 2}, prov))
            self.assertTrue(smoke_reasons("artifacts/v_c/cat_morgulis50.pt", {"steps": 100}, prov))
        with self.assertRaises(ValueError):
            resolve_alpha(1.0, "x.json")
        raw, meta = sidecar_paths("artifacts/carrier/ctrl_g0_10k.jsonl")
        self.assertEqual((raw.name, meta.name), ("ctrl_g0_10k.raw.jsonl", "ctrl_g0_10k.meta.json"))
