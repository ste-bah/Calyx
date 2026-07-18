from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from pypdf import PdfWriter


LAW_DIR = Path(__file__).resolve().parents[1]
if str(LAW_DIR) not in sys.path:
    sys.path.insert(0, str(LAW_DIR))

from authoritative_documents import (  # noqa: E402
    AuthoritativeDocumentError,
    extract_pdf,
    fetch_pdf,
    load_source_contract,
)


SOURCE_CONTRACT = LAW_DIR / "cuyahoga_authoritative_documents.v1.json"


class LiveAuthoritativeDocumentTests(unittest.TestCase):
    """Real-authority tests; no HTTP stubs, fixture PDFs, or mock responses."""

    @classmethod
    def setUpClass(cls):
        cls.contract = load_source_contract(SOURCE_CONTRACT)
        cls.pdfs = {}
        cls.observations = {}
        for opinion_id, document in sorted(cls.contract["by_opinion"].items()):
            storage, storage_observation = fetch_pdf(
                document["storage_url"],
                max_bytes=document["max_pdf_bytes"],
                timeout_seconds=60,
            )
            official, official_observation = fetch_pdf(
                document["download_url"],
                max_bytes=document["max_pdf_bytes"],
                timeout_seconds=60,
            )
            cls.pdfs[opinion_id] = (storage, official)
            cls.observations[opinion_id] = (
                storage_observation,
                official_observation,
            )

    def test_both_live_authorities_return_identical_reviewed_pdf_state(self):
        for opinion_id, document in sorted(self.contract["by_opinion"].items()):
            with self.subTest(opinion_id=opinion_id):
                storage, official = self.pdfs[opinion_id]
                storage_observation, official_observation = self.observations[
                    opinion_id
                ]
                self.assertEqual(storage, official)
                self.assertEqual(storage_observation["sha256"], document["pdf_sha256"])
                self.assertEqual(official_observation["sha256"], document["pdf_sha256"])
                extracted = extract_pdf(storage, document)
                self.assertEqual(extracted["pages"], document["expected_pages"])
                self.assertEqual(
                    extracted["normalized_text_sha256"],
                    document["expected_normalized_text_sha256"],
                )

    def test_real_server_content_length_enforces_maximum_byte_boundary(self):
        document = self.contract["by_opinion"][11227531]
        with self.assertRaises(AuthoritativeDocumentError) as raised:
            fetch_pdf(document["storage_url"], max_bytes=100, timeout_seconds=60)
        self.assertIn("byte limit", str(raised.exception))
        self.assertEqual(raised.exception.context["max_bytes"], 100)

    def test_one_byte_mutation_of_real_pdf_fails_the_reviewed_digest(self):
        document = self.contract["by_opinion"][11227531]
        original = self.pdfs[11227531][0]
        mutated = bytearray(original)
        mutated[len(mutated) // 2] ^= 1
        with self.assertRaises(AuthoritativeDocumentError) as raised:
            extract_pdf(bytes(mutated), document)
        self.assertIn("digest mismatch", str(raised.exception))
        self.assertNotEqual(
            raised.exception.context["actual_sha256"], document["pdf_sha256"]
        )

    def test_truncated_real_pdf_fails_structural_validation(self):
        original = self.pdfs[11227531][0]
        truncated = original[:-4096]
        expected = deepcopy(self.contract["by_opinion"][11227531])
        expected["pdf_sha1"] = hashlib.sha1(truncated).hexdigest()
        expected["pdf_sha256"] = hashlib.sha256(truncated).hexdigest()
        with self.assertRaises(AuthoritativeDocumentError) as raised:
            extract_pdf(truncated, expected)
        self.assertIn("EOF marker", str(raised.exception))

    def test_encrypted_derivative_of_real_pdf_is_explicitly_rejected(self):
        original = self.pdfs[11227531][0]
        writer = PdfWriter()
        writer.append(BytesIO(original))
        writer.encrypt("calyx-real-data-edge")
        destination = BytesIO()
        writer.write(destination)
        encrypted = destination.getvalue()
        expected = deepcopy(self.contract["by_opinion"][11227531])
        expected["pdf_sha1"] = hashlib.sha1(encrypted).hexdigest()
        expected["pdf_sha256"] = hashlib.sha256(encrypted).hexdigest()
        with self.assertRaises(AuthoritativeDocumentError) as raised:
            extract_pdf(encrypted, expected)
        self.assertIn("encrypted", str(raised.exception))

    def test_manifest_cannot_widen_the_reviewed_real_identity_set(self):
        value = json.loads(SOURCE_CONTRACT.read_text(encoding="utf-8"))
        additional = deepcopy(value["documents"][0])
        additional["opinion_id"] = 11227532
        value["documents"].append(additional)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "widened-real-contract.json"
            path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(AuthoritativeDocumentError) as raised:
                load_source_contract(path)
        self.assertIn("not the reviewed byte generation", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
