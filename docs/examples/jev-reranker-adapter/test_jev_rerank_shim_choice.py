#!/usr/bin/env python3
"""Regression tests for jev_rerank_shim_choice probability validation.

Run from this directory:  python3 -m unittest test_jev_rerank_shim_choice -v
Stdlib only, no Jev backend required (HTTP is monkeypatched).
"""
import io
import json
import unittest
import unittest.mock

import jev_rerank_shim_choice as shim


def jev_response(probs):
    return {"answers": {"best": {"probabilities": probs}}}


def rerank_payload(ids=(1, 2, 3)):
    cands = [{"candidate": n, "title": f"page {n}", "text": f"text {n}"}
             for n in ids]
    return {"messages": [
        {"role": "system", "content": shim.RERANK_SYSTEM_PREFIX + " ..."},
        {"role": "user",
         "content": json.dumps({"query": "q", "candidates": cands})},
    ]}


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class ExtractChoiceProbsTest(unittest.TestCase):
    IDS = [1, 2, 3]

    def test_valid(self):
        vals = shim.extract_choice_probs(
            jev_response({"c1": 0.7, "c2": 0.2, "c3": 0.1}), self.IDS)
        self.assertEqual(vals, [0.7, 0.2, 0.1])

    def test_rounding_drift_tolerated(self):
        vals = shim.extract_choice_probs(
            jev_response({"c1": 0.3333, "c2": 0.3333, "c3": 0.3333}), self.IDS)
        self.assertAlmostEqual(sum(vals), 0.9999)

    def test_missing_probabilities_object(self):
        with self.assertRaises(ValueError):
            shim.extract_choice_probs({"answers": {"best": {}}}, self.IDS)

    def test_missing_answers(self):
        with self.assertRaises(ValueError):
            shim.extract_choice_probs({}, self.IDS)

    def test_missing_key(self):
        with self.assertRaisesRegex(ValueError, "missing=.*c3"):
            shim.extract_choice_probs(
                jev_response({"c1": 0.6, "c2": 0.4}), self.IDS)

    def test_extra_key(self):
        with self.assertRaisesRegex(ValueError, "extra=.*c9"):
            shim.extract_choice_probs(
                jev_response({"c1": 0.5, "c2": 0.3, "c3": 0.2, "c9": 0.0}),
                self.IDS)

    def test_bool_rejected(self):
        with self.assertRaises(ValueError):
            shim.extract_choice_probs(
                jev_response({"c1": True, "c2": 0.0, "c3": 1.0}), self.IDS)

    def test_string_rejected(self):
        with self.assertRaises(ValueError):
            shim.extract_choice_probs(
                jev_response({"c1": "0.7", "c2": 0.2, "c3": 0.1}), self.IDS)

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            shim.extract_choice_probs(
                jev_response({"c1": float("nan"), "c2": 0.5, "c3": 0.5}),
                self.IDS)

    def test_inf_rejected(self):
        with self.assertRaises(ValueError):
            shim.extract_choice_probs(
                jev_response({"c1": float("inf"), "c2": 0.0, "c3": 0.0}),
                self.IDS)

    def test_out_of_range_rejected(self):
        for bad in (1.7, -0.2):
            with self.assertRaises(ValueError):
                shim.extract_choice_probs(
                    jev_response({"c1": bad, "c2": 0.0, "c3": 0.0}),
                    self.IDS)

    def test_degenerate_sum_rejected(self):
        with self.assertRaisesRegex(ValueError, "sum"):
            shim.extract_choice_probs(
                jev_response({"c1": 0.9, "c2": 0.9, "c3": 0.9}), self.IDS)
        with self.assertRaisesRegex(ValueError, "sum"):
            shim.extract_choice_probs(
                jev_response({"c1": 0.0, "c2": 0.0, "c3": 0.0}), self.IDS)

    def test_duplicate_candidate_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate candidate"):
            shim.extract_choice_probs(
                jev_response({"c1": 0.6, "c2": 0.4}), [1, 1])

    def test_empty_candidates_rejected(self):
        with self.assertRaises(ValueError):
            shim.extract_choice_probs(jev_response({}), [])


class DuplicateJsonKeyTest(unittest.TestCase):
    def test_duplicate_keys_rejected(self):
        with self.assertRaises(ValueError):
            json.loads('{"c1": 0.6, "c1": 0.4}',
                       object_pairs_hook=shim._reject_dupes)

    def test_normal_json_accepted(self):
        d = json.loads('{"c1": 0.6, "c2": 0.4}',
                       object_pairs_hook=shim._reject_dupes)
        self.assertEqual(d, {"c1": 0.6, "c2": 0.4})


class JevRerankEndToEndTest(unittest.TestCase):
    def run_shim(self, response_json):
        fake = FakeResp(json.dumps(response_json).encode())
        with unittest.mock.patch.object(shim.urllib.request, "urlopen",
                                        return_value=fake):
            return shim.jev_rerank(rerank_payload())

    def test_valid_response_maps_scores(self):
        out = self.run_shim(jev_response({"c1": 0.1, "c2": 0.7, "c3": 0.2}))
        self.assertEqual(out, {"scores": [
            {"candidate": 1, "relevance": 0.1},
            {"candidate": 2, "relevance": 0.7},
            {"candidate": 3, "relevance": 0.2},
        ]})

    def test_malformed_response_raises(self):
        # Old behaviour silently fabricated 0.0 for missing keys; now the
        # exception propagates so the handler answers 500 and ai-memory
        # keeps its original order.
        with self.assertRaises(ValueError):
            self.run_shim(jev_response({"c1": 0.9}))

    def test_200_with_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            self.run_shim({"answers": {"best": {"choice": "c1"}}})


if __name__ == "__main__":
    unittest.main()
