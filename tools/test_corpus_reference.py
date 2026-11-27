import contextlib
import hashlib
import io
import json
import unicodedata
import unittest
from fractions import Fraction

from corpus_reference import (
    MAX_RECORDS,
    MAX_RECORD_BYTES,
    SYNTHETIC_APPROVAL,
    ClusterResult,
    CorpusError,
    MemberInfo,
    Record,
    SourceApproval,
    audit_splits,
    build_allowlist,
    build_report,
    canonical_digest,
    cluster_identifier,
    decode_text,
    demo_report,
    jaccard,
    main,
    near_duplicate,
    normalized_sha256,
    split_bucket,
    split_for_cluster,
    validate_records,
    word_shingles,
)


def source(source_id="src-a", revision="rev-1", approval=SYNTHETIC_APPROVAL):
    return SourceApproval(source_id, revision, approval)


def rec(
    record_id,
    text,
    *,
    source_id="src-a",
    revision="rev-1",
    group=None,
    language="en",
    domain="prose",
):
    return Record(
        record_id,
        source_id,
        revision,
        group if group is not None else f"grp/{record_id}",
        language,
        domain,
        text,
    )


def report(records, *, allowlist=None, **kwargs):
    return build_report(records, allowlist or (source(),), **kwargs)


def cluster_for(result, record_id):
    for cluster in result["clusters"]:
        if any(member["record_id"] == record_id for member in cluster["members"]):
            return cluster
    raise AssertionError(f"record not found in any cluster: {record_id}")


class SourceAllowlistTests(unittest.TestCase):
    def test_allowlist_fail_closed(self):
        with self.assertRaises(CorpusError):
            build_allowlist(())
        with self.assertRaises(CorpusError):
            build_allowlist(None)
        with self.assertRaises(CorpusError):
            build_allowlist((source("src", ""),))
        with self.assertRaises(CorpusError):
            build_allowlist((source("src", "rev", "pending"),))
        with self.assertRaises(CorpusError):
            build_allowlist((source("src", "rev", "denied"),))
        with self.assertRaises(CorpusError):
            build_allowlist((source("src"), source("src")))
        allowlist = build_allowlist((source(),))
        with self.assertRaises(CorpusError):
            validate_records((rec("a", "text", source_id="unknown"),), allowlist)
        with self.assertRaises(CorpusError):
            validate_records((rec("a", "text", revision="other"),), allowlist)
        with self.assertRaises(CorpusError):
            validate_records(
                (Record("a", "src-a", "", "grp/a", "en", "prose", "text"),),
                allowlist,
            )


class RecordValidationTests(unittest.TestCase):
    def setUp(self):
        self.allowlist = build_allowlist((source(),))

    def test_malformed_records_fail_closed(self):
        bad = (
            rec("", "text"),
            rec("a", ""),
            rec("a", "text", language="de"),
            rec("a", "text", domain="code"),
            rec("a", "text", domain="math"),
            rec("a", "text", group=""),
            rec("a", "text", source_id=""),
            rec("a", "\ud800"),
            rec("a", "a" * (MAX_RECORD_BYTES + 1)),
            Record("a", "src-a", "rev-1", "grp/a", "en", "prose", None),
            Record(None, "src-a", "rev-1", "grp/a", "en", "prose", "text"),
        )
        for record in bad:
            with self.assertRaises(CorpusError, msg=repr(record)):
                validate_records((record,), self.allowlist)
        with self.assertRaises(CorpusError):
            validate_records((rec("a", "one"), rec("a", "two")), self.allowlist)
        with self.assertRaises(CorpusError):
            validate_records(None, self.allowlist)

    def test_capacity_and_byte_boundaries(self):
        many = tuple(
            rec(f"r{index}", f"text {index}", group=f"grp/{index}")
            for index in range(MAX_RECORDS)
        )
        self.assertEqual(len(validate_records(many, self.allowlist)), MAX_RECORDS)
        too_many = many + (rec("extra", "text", group="grp/extra"),)
        with self.assertRaises(CorpusError):
            validate_records(too_many, self.allowlist)
        self.assertEqual(
            len(validate_records((rec("max", "a" * MAX_RECORD_BYTES),), self.allowlist)),
            1,
        )
        with self.assertRaises(CorpusError):
            validate_records(
                (rec("over", "a" * (MAX_RECORD_BYTES + 1)),),
                self.allowlist,
            )

    def test_utf8_decode_helper(self):
        self.assertEqual(decode_text("Zażółć".encode("utf-8")), "Zażółć")
        with self.assertRaises(CorpusError):
            decode_text(b"\xff")
        with self.assertRaises(CorpusError):
            decode_text("not bytes")

    def test_empty_corpus_is_valid(self):
        result = report(())
        self.assertEqual(result["stats"]["raw"]["records"], 0)
        self.assertEqual(result["clusters"], [])
        self.assertTrue(result["audit"]["ok"])
        with self.assertRaises(CorpusError):
            build_report((), ())


class PreservationAndHashTests(unittest.TestCase):
    def test_text_is_preserved_and_hashes_are_independent(self):
        text = "  Zażółć\tgęślą\r\njaźń  "
        validated = validate_records(
            (rec("a", text),),
            build_allowlist((source(),)),
        )[0]
        self.assertEqual(validated.record.text, text)
        self.assertEqual(validated.byte_length, len(text.encode("utf-8")))
        self.assertEqual(
            validated.raw_sha256,
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
        expected = unicodedata.normalize(
            "NFC", text.replace("\r\n", "\n").replace("\r", "\n")
        )
        self.assertEqual(
            validated.normalized_sha256,
            hashlib.sha256(expected.encode("utf-8")).hexdigest(),
        )
        self.assertNotEqual(validated.raw_sha256, validated.normalized_sha256)

    def test_nfc_and_crlf_duplicates(self):
        result = report(
            (
                rec("b", "Cafe\u0301\r\nline", group="grp/b"),
                rec("a", "Café\nline", group="grp/a"),
            )
        )
        self.assertEqual(len(result["clusters"]), 1)
        cluster = result["clusters"][0]
        self.assertEqual(
            [member["record_id"] for member in cluster["members"]], ["a", "b"]
        )
        self.assertEqual(cluster["representative"], "a")
        self.assertTrue(
            all(
                "exact_normalized" in member["reasons"]
                for member in cluster["members"]
            )
        )
        self.assertNotEqual(
            cluster["members"][0]["raw_sha256"],
            cluster["members"][1]["raw_sha256"],
        )

    def test_no_casefold_or_diacritic_stripping(self):
        self.assertNotEqual(
            normalized_sha256("Żółć gęślą jaźń."),
            normalized_sha256("Zolc gesla jazn."),
        )
        result = report(
            (
                rec("a", "Żółć gęślą jaźń."),
                rec("b", "Zolc gesla jazn."),
            )
        )
        self.assertEqual(len(result["clusters"]), 2)

    def test_exact_raw_and_normalized_reasons(self):
        result = report((rec("a", "same text"), rec("b", "same text")))
        self.assertEqual(len(result["clusters"]), 1)
        for member in result["clusters"][0]["members"]:
            self.assertIn("exact_raw", member["reasons"])
            self.assertIn("exact_normalized", member["reasons"])


class ClusteringTests(unittest.TestCase):
    def test_group_union_across_nonidentical_texts_and_sources(self):
        result = report(
            (
                rec("a", "one two three", group="shared", source_id="src-a"),
                rec("b", "four five six", group="shared", source_id="src-b"),
            ),
            allowlist=(source("src-a"), source("src-b")),
        )
        self.assertEqual(len(result["clusters"]), 1)
        self.assertEqual(
            [member["reasons"] for member in result["clusters"][0]["members"]],
            [["source_group"], ["source_group"]],
        )
        self.assertEqual(result["near_duplicate_pairs"], [])

    def test_near_threshold_is_inclusive(self):
        tokens = [f"t{index}" for index in range(1, 10)]
        a_text = " ".join(tokens[:8])
        b_text = " ".join(tokens)
        a_shingles = word_shingles(a_text)
        b_shingles = word_shingles(b_text)
        expected_a = {tuple(tokens[index:index + 5]) for index in range(4)}
        expected_b = {tuple(tokens[index:index + 5]) for index in range(5)}
        self.assertEqual(a_shingles, frozenset(expected_a))
        self.assertEqual(b_shingles, frozenset(expected_b))
        self.assertEqual(jaccard(a_shingles, b_shingles), Fraction(4, 5))
        self.assertTrue(near_duplicate(a_shingles, b_shingles))
        result = report((rec("a", a_text), rec("b", b_text)))
        self.assertEqual(len(result["clusters"]), 1)
        self.assertTrue(
            all(
                "near_duplicate" in member["reasons"]
                for member in result["clusters"][0]["members"]
            )
        )

    def test_near_below_threshold_is_rejected(self):
        tokens = [f"u{index}" for index in range(1, 9)]
        a_text = " ".join(tokens[:7])
        b_text = " ".join(tokens)
        a_shingles = word_shingles(a_text)
        b_shingles = word_shingles(b_text)
        expected_a = {tuple(tokens[index:index + 5]) for index in range(3)}
        expected_b = {tuple(tokens[index:index + 5]) for index in range(4)}
        self.assertEqual(a_shingles, frozenset(expected_a))
        self.assertEqual(b_shingles, frozenset(expected_b))
        self.assertEqual(jaccard(a_shingles, b_shingles), Fraction(3, 4))
        self.assertFalse(near_duplicate(a_shingles, b_shingles))
        result = report((rec("a", a_text), rec("b", b_text)))
        self.assertEqual(len(result["clusters"]), 2)

    def test_short_documents_have_no_near_edges(self):
        left = "s1 s2 s3 s4"
        right = "s1 s2 s3 s5"
        self.assertEqual(word_shingles(left), frozenset())
        self.assertFalse(near_duplicate(word_shingles(left), word_shingles(right)))
        result = report((rec("a", left), rec("b", right)))
        self.assertEqual(len(result["clusters"]), 2)

    def test_transitive_closure_across_exact_group_and_near_edges(self):
        tokens = [f"v{index}" for index in range(1, 11)]
        a_text = " ".join(tokens[:8])
        b_text = " ".join(tokens[:9])
        c_text = " ".join(tokens)
        result = report((rec("a", a_text), rec("b", b_text), rec("c", c_text)))
        self.assertEqual(len(result["clusters"]), 1)
        self.assertEqual(
            [member["record_id"] for member in result["clusters"][0]["members"]],
            ["a", "b", "c"],
        )
        self.assertEqual(
            result["near_duplicate_pairs"],
            [["a", "b"], ["b", "c"]],
        )
        self.assertFalse(near_duplicate(word_shingles(a_text), word_shingles(c_text)))

    def test_transitive_closure_across_source_ids_and_languages(self):
        result = report(
            (
                rec(
                    "a",
                    "x1 x2 x3 x4 x5 x6 x7 x8",
                    source_id="src-a",
                    language="en",
                    group="g1",
                ),
                rec(
                    "b",
                    "x1 x2 x3 x4 x5 x6 x7 x8 x9",
                    source_id="src-b",
                    language="pl",
                    group="g2",
                ),
                rec(
                    "c",
                    "café group edge text",
                    source_id="src-c",
                    language="en",
                    group="g2",
                ),
                rec(
                    "d",
                    "cafe\u0301 group edge text",
                    source_id="src-a",
                    language="pl",
                    group="g3",
                ),
            ),
            allowlist=(source("src-a"), source("src-b"), source("src-c")),
        )
        self.assertEqual(len(result["clusters"]), 1)
        reasons = {
            member["record_id"]: member["reasons"]
            for member in result["clusters"][0]["members"]
        }
        self.assertIn("near_duplicate", reasons["a"])
        self.assertIn("near_duplicate", reasons["b"])
        self.assertIn("source_group", reasons["b"])
        self.assertIn("source_group", reasons["c"])
        self.assertIn("exact_normalized", reasons["c"])
        self.assertIn("exact_normalized", reasons["d"])


class QuarantineTests(unittest.TestCase):
    def test_benchmark_exact_normalized_quarantines_whole_cluster(self):
        benchmark = "Cafe\u0301 benchmark question\nwith two lines."
        result = report(
            (
                rec(
                    "a",
                    "Café benchmark question\r\nwith two lines.",
                    group="grp/a",
                ),
                rec("b", "Another member of the benchmark family.", group="grp/a"),
                rec(
                    "c",
                    "Café benchmark question\nwith two lines!",
                    group="grp/c",
                ),
            ),
            benchmark_exclusions=(benchmark,),
        )
        cluster_a = cluster_for(result, "a")
        self.assertIsNone(cluster_a["split"])
        self.assertEqual(
            cluster_a["quarantine_reasons"],
            ["benchmark_exact_normalized"],
        )
        self.assertEqual(
            [member["record_id"] for member in cluster_a["members"]],
            ["a", "b"],
        )
        cluster_c = cluster_for(result, "c")
        self.assertIsNotNone(cluster_c["split"])
        self.assertNotIn("benchmark_exact_normalized", cluster_c["quarantine_reasons"])
        self.assertNotIn(
            cluster_a["cluster_id"],
            result["splits"]["train"]
            + result["splits"]["validation"]
            + result["splits"]["test"],
        )

    def test_excluded_record_id_quarantines_whole_cluster(self):
        records = (
            rec("a", "first member text", group="grp/a"),
            rec("b", "second member text", group="grp/a"),
            rec("c", "unrelated text", group="grp/c"),
        )
        result = report(records, excluded_record_ids=("a",))
        cluster_a = cluster_for(result, "a")
        self.assertIsNone(cluster_a["split"])
        self.assertEqual(cluster_a["quarantine_reasons"], ["excluded_record_id"])
        self.assertEqual(
            [member["record_id"] for member in cluster_a["members"]],
            ["a", "b"],
        )
        self.assertIsNotNone(cluster_for(result, "c")["split"])
        unknown = report(records, excluded_record_ids=("missing",))
        self.assertEqual(unknown["stats"]["quarantined"]["records"], 0)
        self.assertEqual(unknown["stats"]["excluded_record_ids_matched"], 0)

    def test_benchmark_and_excluded_reasons_combine(self):
        result = report(
            (
                rec("a", "Café benchmark line", group="grp/a"),
                rec("b", "other member", group="grp/a"),
            ),
            benchmark_exclusions=("Cafe\u0301 benchmark line",),
            excluded_record_ids=("b",),
        )
        self.assertEqual(
            result["clusters"][0]["quarantine_reasons"],
            ["benchmark_exact_normalized", "excluded_record_id"],
        )


class AuditTests(unittest.TestCase):
    def test_no_split_leakage(self):
        tokens = [f"w{index}" for index in range(1, 10)]
        result = report(
            (
                rec("a", " ".join(tokens[:8]), group="g1"),
                rec("b", " ".join(tokens), group="g2"),
                rec("c", "completely different words here", group="g3"),
                rec("d", "completely different words here", group="g4"),
            ),
            seed="audit-seed",
            train_pct=50,
            val_pct=25,
        )
        self.assertTrue(result["audit"]["ok"])
        for name, value in result["audit"]["checks"].items():
            self.assertTrue(value, name)
        split_sets = {
            name: set(result["splits"][name])
            for name in ("train", "validation", "test")
        }
        for left in split_sets:
            for right in split_sets:
                if left < right:
                    self.assertFalse(split_sets[left] & split_sets[right])
        for left, right in result["near_duplicate_pairs"]:
            self.assertEqual(
                cluster_for(result, left)["cluster_id"],
                cluster_for(result, right)["cluster_id"],
            )
        self.assertEqual(
            sum(len(result["splits"][name]) for name in split_sets),
            result["stats"]["eligible_clusters"],
        )

    def test_audit_detects_cross_split_fingerprint_leak(self):
        def member(record_id, normalized, group):
            return MemberInfo(
                record_id,
                "src-a",
                "rev-1",
                group,
                "en",
                "prose",
                4,
                "raw-" + record_id,
                normalized,
                (),
            )

        left = ClusterResult(
            "c1",
            (member("a", "same-fingerprint", "g1"),),
            "a",
            "same-fingerprint",
            (),
            "train",
            (),
        )
        right = ClusterResult(
            "c2",
            (member("b", "same-fingerprint", "g2"),),
            "b",
            "same-fingerprint",
            (),
            "test",
            (),
        )
        audit = audit_splits((left, right), ())
        self.assertFalse(audit["ok"])
        self.assertFalse(audit["checks"]["normalized_fingerprints_disjoint"])

    def test_audit_detects_cross_split_group_leak(self):
        def member(record_id, normalized, group):
            return MemberInfo(
                record_id,
                "src-a",
                "rev-1",
                group,
                "en",
                "prose",
                4,
                "raw-" + record_id,
                normalized,
                (),
            )

        left = ClusterResult(
            "c1",
            (member("a", "n1", "shared-group"),),
            "a",
            "n1",
            (),
            "train",
            (),
        )
        right = ClusterResult(
            "c2",
            (member("b", "n2", "shared-group"),),
            "b",
            "n2",
            (),
            "test",
            (),
        )
        audit = audit_splits((left, right), ())
        self.assertFalse(audit["ok"])
        self.assertFalse(audit["checks"]["source_groups_disjoint"])


class DeterminismAndStatsTests(unittest.TestCase):
    def test_representative_ordering_and_shuffled_input(self):
        records = (
            rec("z", "shared text", group="g1"),
            rec("a", "shared text", group="g2"),
            rec("m", "shared text", group="g3"),
        )
        first = report(records, seed="det", train_pct=50, val_pct=25)
        second = report(tuple(reversed(records)), seed="det", train_pct=50, val_pct=25)
        self.assertEqual(first, second)
        cluster = first["clusters"][0]
        self.assertEqual(cluster["representative"], "a")
        self.assertEqual(
            [member["record_id"] for member in cluster["members"]],
            ["a", "m", "z"],
        )
        self.assertEqual(
            cluster["cluster_id"],
            hashlib.sha256(
                json.dumps(["a", "m", "z"], separators=(",", ":")).encode()
            ).hexdigest(),
        )
        self.assertEqual(cluster["cluster_id"], cluster_identifier(["z", "a", "m"]))
        self.assertEqual(first["digest"], canonical_digest(first))
        self.assertEqual(len(first["digest"]), 64)
        self.assertNotIn("timestamp", json.dumps(first))

    def test_split_bucket_and_invalid_config(self):
        for cluster_id in ("0" * 64, "f" * 64, "a" * 64):
            expected = int.from_bytes(
                hashlib.sha256(
                    ("seed\0" + cluster_id).encode("utf-8")
                ).digest()[:8],
                "big",
            ) % 100
            self.assertEqual(split_bucket(cluster_id, "seed"), expected)
            self.assertIn(
                split_for_cluster(cluster_id, "seed", 50, 25),
                ("train", "validation", "test"),
            )
        with self.assertRaises(CorpusError):
            split_for_cluster("a" * 64, "", 98, 1)
        for train_pct, val_pct in (
            (0, 1),
            (100, 1),
            (98, 2),
            (50, 50),
            (-1, 1),
            (98, 0),
            (True, 1),
            (98.0, 1),
        ):
            with self.assertRaises(CorpusError):
                split_for_cluster("a" * 64, "seed", train_pct, val_pct)

    def test_stats_counts_bytes_and_reasons(self):
        records = (
            rec("a", "Café text", group="g1", language="en"),
            rec("b", "Cafe\u0301 text", group="g2", language="en"),
            rec("c", "Żółć", group="g3", language="pl"),
            rec("d", "quarantined member", group="g4", language="pl"),
            rec("e", "quarantined member", group="g4", language="pl"),
        )
        result = report(records, excluded_record_ids=("d",))
        stats = result["stats"]
        self.assertEqual(stats["raw"]["records"], 5)
        self.assertEqual(
            stats["raw"]["bytes"],
            sum(len(record.text.encode("utf-8")) for record in records),
        )
        self.assertEqual(stats["retained"]["records"], 2)
        self.assertEqual(stats["quarantined"]["records"], 2)
        self.assertEqual(
            stats["quarantined"]["reasons"],
            {"excluded_record_id": 1},
        )
        self.assertEqual(stats["duplicate_records_removed"], 1)
        self.assertEqual(stats["duplicate_records_removed_including_quarantine"], 3)
        self.assertEqual(stats["raw"]["languages"]["en"]["records"], 2)
        self.assertEqual(stats["raw"]["languages"]["pl"]["records"], 3)
        self.assertEqual(stats["retained"]["languages"]["pl"]["records"], 1)


class DemoTests(unittest.TestCase):
    def test_demo_is_deterministic_and_covers_scenarios(self):
        first = demo_report()
        second = demo_report()
        self.assertEqual(first, second)
        self.assertEqual(first["stats"]["raw"]["records"], 20)
        self.assertEqual(len(first["clusters"]), 13)
        self.assertEqual(first["stats"]["quarantined_clusters"], 2)
        self.assertEqual(first["stats"]["quarantined"]["records"], 4)
        self.assertEqual(first["stats"]["retained"]["records"], 11)
        self.assertEqual(len(first["inputs"]), 20)
        self.assertTrue(any("\r\n" in item["text"] for item in first["inputs"]))
        self.assertTrue(any("\u0301" in item["text"] for item in first["inputs"]))
        self.assertTrue(any("Żółć" in item["text"] for item in first["inputs"]))
        reasons = {
            reason
            for cluster in first["clusters"]
            for member in cluster["members"]
            for reason in member["reasons"]
        }
        self.assertTrue(
            {"exact_normalized", "source_group", "near_duplicate"} <= reasons
        )
        quarantine_reasons = {
            reason
            for cluster in first["clusters"]
            for reason in cluster["quarantine_reasons"]
        }
        self.assertEqual(
            quarantine_reasons,
            {"benchmark_exact_normalized", "excluded_record_id"},
        )
        self.assertTrue(first["audit"]["ok"])
        self.assertEqual(
            sum(
                len(first["splits"][name])
                for name in ("train", "validation", "test")
            ),
            11,
        )
        self.assertEqual(len(first["near_duplicate_pairs"]), 4)
        self.assertEqual(first["digest"], canonical_digest(first))
        self.assertEqual(len(first["digest"]), 64)

    def test_cli_demo(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["--demo"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue()), demo_report())
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            self.assertEqual(main([]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
