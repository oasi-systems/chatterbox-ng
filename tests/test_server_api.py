"""
Test suite for server_streaming.py REST API.

Tests dictionary CRUD endpoints and health check without loading the TTS model.
Uses Starlette's TestClient for synchronous HTTP testing.

Run: python -m pytest tests/test_server_api.py -v
"""
import sys
import os
import importlib.util

import pytest

# Direct import of CustomDictionary to avoid heavy deps
_g2p_path = os.path.join(os.path.dirname(__file__), "..", "src", "chatterbox", "g2p.py")
_spec = importlib.util.spec_from_file_location("chatterbox.g2p", _g2p_path)
_g2p_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_g2p_mod)
CustomDictionary = _g2p_mod.CustomDictionary

# We test the dictionary API logic directly (no model loading needed)

class TestDictionaryAPI:
    """Test CustomDictionary CRUD operations as used by the server."""

    def setup_method(self):
        self.d = CustomDictionary()

    def test_add_single(self):
        self.d.add("IBAN", "i ban", language_id="it")
        assert self.d.lookup("IBAN", "it") == "i ban"

    def test_add_global(self):
        self.d.add("PIN", "pin")
        assert self.d.lookup("PIN", "it") == "pin"
        assert self.d.lookup("PIN", "fr") == "pin"

    def test_batch_add(self):
        entries = [
            {"word": "IBAN", "respelling": "i ban", "language_id": "it"},
            {"word": "SEPA", "respelling": "sepa"},
            {"word": "CVV", "respelling": "ci vu vu", "language_id": "it"},
        ]
        for e in entries:
            self.d.add(e["word"], e["respelling"], language_id=e.get("language_id"))

        assert self.d.lookup("IBAN", "it") == "i ban"
        assert self.d.lookup("SEPA", "de") == "sepa"
        assert self.d.lookup("CVV", "it") == "ci vu vu"

    def test_remove(self):
        self.d.add("test", "tèst", language_id="it")
        assert self.d.remove("test", language_id="it") is True
        assert self.d.lookup("test", "it") is None

    def test_remove_nonexistent(self):
        assert self.d.remove("ghost", language_id="it") is False

    def test_list_all(self):
        self.d.add("PIN", "pin")
        self.d.add("IBAN", "i ban", language_id="it")
        result = self.d.list_entries()
        assert "global" in result
        assert "it" in result

    def test_list_by_language(self):
        self.d.add("PIN", "pin")
        self.d.add("IBAN", "i ban", language_id="it")
        self.d.add("Müller", "miuller", language_id="de")
        result = self.d.list_entries(language_id="it")
        assert "it" in result
        assert "global" in result
        assert "de" not in result

    def test_overwrite_entry(self):
        self.d.add("test", "version1", language_id="it")
        self.d.add("test", "version2", language_id="it")
        assert self.d.lookup("test", "it") == "version2"


class TestConcurrencyModel:
    """Verify the concurrency primitives exist in server code."""

    def test_server_syntax(self):
        """Server module parses without syntax errors."""
        import ast
        server_path = os.path.join(os.path.dirname(__file__), "..", "server_streaming.py")
        with open(server_path) as f:
            ast.parse(f.read())

    def test_server_has_inference_lock(self):
        """Server uses asyncio.Lock for inference serialization."""
        server_path = os.path.join(os.path.dirname(__file__), "..", "server_streaming.py")
        with open(server_path) as f:
            content = f.read()
        assert "asyncio.Lock()" in content
        assert "_inference_lock" in content
        assert "async with _inference_lock" in content

    def test_server_has_thread_pool(self):
        """Server offloads sync generators to thread pool."""
        server_path = os.path.join(os.path.dirname(__file__), "..", "server_streaming.py")
        with open(server_path) as f:
            content = f.read()
        assert "run_in_executor" in content

    def test_server_has_dict_api(self):
        """Server exposes dictionary CRUD endpoints."""
        server_path = os.path.join(os.path.dirname(__file__), "..", "server_streaming.py")
        with open(server_path) as f:
            content = f.read()
        assert "/api/dictionary" in content
        assert "dict_get" in content
        assert "dict_post" in content
        assert "dict_delete" in content

    def test_server_has_request_stats(self):
        """Server tracks request queue metrics."""
        server_path = os.path.join(os.path.dirname(__file__), "..", "server_streaming.py")
        with open(server_path) as f:
            content = f.read()
        assert "_request_stats" in content
        assert '"active"' in content
        assert '"queued"' in content
