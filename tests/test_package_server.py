"""Source-export safety tests; standard library only, no model/network calls."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import package_server as packager


class PackageServerTests(unittest.TestCase):
    def test_includes_current_deployment_sources(self):
        for name in ("compose.low-memory.yaml", "API_ONLY_DEPLOYMENT.md", "SYNC_GUIDE.md",
                     "infra/deploy/login_guard.lua", "back/core/features.py", ".env.example",
                     "infra/deploy/.env.example", "evaluation/results/.gitkeep"):
            with self.subTest(name=name):
                self.assertTrue(packager.allowed_path(name))

    def test_excludes_secrets_and_runtime_artifacts(self):
        for name in (".env.production", "deployment-secrets.txt", "deployment-acceptance-state.json",
                     "deployment-browser-chat.png", "backend-build-metadata.json",
                     "domain-change-20260919/private.before", "security-change-20260919/private.before",
                     "back/.env.local", "infra/deploy/private.pem", "back/sessions.db",
                     "infra/deploy/deployment-secrets.txt", "back/model.safetensors",
                     "back/sessions.db-wal", "back/cache.sqlite3", "front/node_modules/foo.js",
                     "back/__pycache__/api.pyc", "evaluation/results/private.json", "output/project.zip"):
            with self.subTest(name=name):
                self.assertFalse(packager.allowed_path(name))

    def test_rejects_unsafe_paths(self):
        for name in ("", "/back/api.py", "back/../../secret.txt", "back\\..\\secret.txt"):
            with self.subTest(name=name):
                self.assertFalse(packager.allowed_path(name))

    def test_unversioned_source_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(packager, "ROOT", Path(directory)):
            with self.assertRaises(SystemExit):
                packager.source_paths()

    def test_unversioned_collection_keeps_source_only(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(packager, "ROOT", Path(directory)):
            root = Path(directory)
            (root / "back").mkdir()
            (root / "back/api.py").write_text("pass\n")
            (root / "back/.env.local").write_text("PRIVATE=fixture\n")
            (root / ".env.production").write_text("PRIVATE=fixture\n")
            (root / "compose.low-memory.yaml").write_text("services: {}\n")
            names, revision = packager.source_paths(True)
            self.assertIsNone(revision)
            self.assertEqual(set(packager.collect_files(names)), {"back/api.py", "compose.low-memory.yaml"})

    def test_linked_file_is_refused(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(packager, "ROOT", Path(directory)):
            root = Path(directory)
            (root / "back").mkdir()
            (root / "private.txt").write_text("private fixture")
            (root / "back/api.py").symlink_to(root / "private.txt")
            with self.assertRaises(SystemExit):
                packager.collect_files({"back/api.py"})

    def test_linked_top_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(packager, "ROOT", Path(directory)):
            root = Path(directory)
            (root / "private").mkdir()
            (root / "back").symlink_to(root / "private", target_is_directory=True)
            with self.assertRaises(SystemExit):
                packager.source_paths(True)

    def test_shell_newlines_are_normalized(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(packager, "ROOT", Path(directory)):
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "scripts/test.sh").write_bytes(b"\xef\xbb\xbf#!/bin/sh\r\necho ok\r\n")
            self.assertEqual(packager.collect_files({"scripts/test.sh"})["scripts/test.sh"],
                             b"#!/bin/sh\necho ok\n")


if __name__ == "__main__":
    unittest.main()
