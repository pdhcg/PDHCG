# Copyright 2026 Hongpei Li
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for the license checker; no solver build is needed."""

import io
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from zipfile import ZipFile

try:
    from . import check_license
except ImportError:
    import check_license


class LicenseChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="pdhcg-license-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.license = (Path(__file__).resolve().parents[1] / "LICENSE").read_bytes()
        (self.root / "LICENSE").write_bytes(self.license)
        (self.root / "pyproject.toml").write_text(
            '[project]\nname = "pdhcg"\nlicense = {text = "Apache-2.0"}\n', encoding="utf-8"
        )
        (self.root / "src").mkdir()
        self.source = self.root / "src" / "sample.c"
        self.source.write_text("/* SPDX-License-Identifier: Apache-2.0 */\n", encoding="utf-8")

    def package(self, wheel=True, include_license=True, license_text=None, license_id="Apache-2.0"):
        prefix = "pdhcg-0.3.1.dist-info" if wheel else "pdhcg-0.3.1"
        metadata = f"Name: pdhcg\nVersion: 0.3.1\nLicense: {license_id}\n".encode()
        files = {prefix + ("/METADATA" if wheel else "/PKG-INFO"): metadata}
        if include_license:
            files[prefix + ("/licenses/LICENSE" if wheel else "/LICENSE")] = (
                self.license if license_text is None else license_text
            )
        path = self.root / ("sample.whl" if wheel else "sample.tar.gz")
        if wheel:
            with ZipFile(path, "w") as archive:
                for name, contents in files.items():
                    archive.writestr(name, contents)
        else:
            with tarfile.open(path, "w:gz") as archive:
                for name, contents in files.items():
                    member = tarfile.TarInfo(name)
                    member.size = len(contents)
                    archive.addfile(member, io.BytesIO(contents))
        return path

    def test_source_notices(self):
        for header in [
            "/* SPDX-License-Identifier: Apache-2.0 */",
            "# SPDX-License-Identifier: Apache-2.0",
            "/* Copyright 2026 Example. Licensed under the Apache License, Version 2.0. */",
        ]:
            with self.subTest(header=header):
                self.assertTrue(check_license.has_apache_notice(header, Path("sample.py")))
        for header in ["int x;", "// SPDX-License-Identifier: MIT", "// SPDX-License-Identifier: Apache-2.0 OR MIT"]:
            with self.subTest(header=header):
                self.assertFalse(check_license.has_apache_notice(header, Path("sample.py")))

    def test_only_leading_comments_are_notices(self):
        self.assertFalse(check_license.has_apache_notice(
            'license = "SPDX-License-Identifier: Apache-2.0"', Path("sample.py")
        ))
        source = '# SPDX-License-Identifier: Apache-2.0\nlicense = "SPDX-License-Identifier: MIT"'
        self.assertTrue(check_license.has_apache_notice(source, Path("sample.py")))
        macro = '#define LICENSE_TEXT "Licensed under the Apache License, Version 2.0"'
        self.assertFalse(check_license.has_apache_notice(macro, Path("sample.c")))
        after_code = 'int x;\n// SPDX-License-Identifier: Apache-2.0'
        self.assertFalse(check_license.has_apache_notice(after_code, Path("sample.c")))

    def test_conflicting_header_licenses(self):
        for text in [
            "/* SPDX-License-Identifier: Apache-2.0 */ /* SPDX-License-Identifier: MIT */",
            "/* Licensed under the Apache License, Version 2.0 */\n" + "// comment\n" * 45
            + "// SPDX-License-Identifier: MIT\n",
        ]:
            with self.subTest(text=text):
                self.assertFalse(check_license.has_apache_notice(text, Path("sample.c")))

    def test_missing_source_notice(self):
        self.assertEqual(check_license.check_repository(self.root), [])
        self.source.write_text("int x;\n", encoding="utf-8")
        self.assertTrue(any("sample.c" in error for error in check_license.check_repository(self.root)))

    def test_python_stub_is_checked(self):
        (self.root / "pdhcg").mkdir()
        (self.root / "pdhcg" / "core.pyi").write_text("def solve() -> None: ...\n", encoding="utf-8")
        self.assertTrue(any("core.pyi" in error for error in check_license.check_repository(self.root)))

    def test_tracked_ignored_source_is_checked(self):
        (self.root / "test").mkdir()
        ignored = self.root / "test" / "ignored.py"
        ignored.write_text("def solve(): pass\n", encoding="utf-8")
        (self.root / ".gitignore").write_text("test/\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(self.root)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(self.root), "add", "-f", "test/ignored.py"], check=True, capture_output=True)
        self.assertTrue(any("ignored.py" in error for error in check_license.check_repository(self.root)))
        ignored.unlink()
        self.assertEqual(check_license.check_repository(self.root), [])

    def test_root_license_tampering(self):
        (self.root / "LICENSE").write_bytes(self.license.replace(b"Haihao Lu", b"Someone else"))
        self.assertTrue(any("attribution changed" in error for error in check_license.check_repository(self.root)))

    def test_project_metadata_mismatch(self):
        (self.root / "pyproject.toml").write_text('[project]\nlicense = "MIT"\n', encoding="utf-8")
        self.assertTrue(any("project.license" in error for error in check_license.check_repository(self.root)))

    def test_package_metadata_mismatch(self):
        for wheel in [True, False]:
            with self.subTest(wheel=wheel):
                package = self.package(wheel=wheel, license_id="MIT")
                errors = check_license.check_artifact(package, self.license)
                self.assertTrue(any("metadata must identify" in error for error in errors))

    def test_missing_package_license(self):
        for wheel in [True, False]:
            with self.subTest(wheel=wheel):
                package = self.package(wheel=wheel, include_license=False)
                errors = check_license.check_artifact(package, self.license)
                self.assertTrue(any("missing packaged LICENSE" in error for error in errors))

    def test_unrelated_license_does_not_cover_package(self):
        for wheel in [True, False]:
            with self.subTest(wheel=wheel):
                metadata_name = "pdhcg-0.3.1.dist-info/METADATA" if wheel else "pdhcg-0.3.1/PKG-INFO"
                files = {
                    metadata_name: b"Name: pdhcg\nLicense: Apache-2.0\n",
                    "unrelated/LICENSE": self.license,
                }
                errors = check_license.check_archive_files(files, files.__getitem__, "package", self.license, wheel)
                self.assertTrue(any("missing packaged LICENSE" in error for error in errors))

    def test_mismatched_package_license(self):
        for wheel in [True, False]:
            with self.subTest(wheel=wheel):
                package = self.package(wheel=wheel, license_text=b"Apache-2.0\n")
                errors = check_license.check_artifact(package, self.license)
                self.assertTrue(any("differs from repository LICENSE" in error for error in errors))

    def test_valid_packages(self):
        for wheel in [True, False]:
            with self.subTest(wheel=wheel):
                package = self.package(wheel=wheel)
                self.assertEqual(check_license.check_artifact(package, self.license), [])

    def test_license_expression_metadata(self):
        metadata = b"Name: pdhcg\nLicense-Expression: Apache-2.0\n"
        self.assertEqual(check_license.check_metadata(metadata, "METADATA"), [])
        conflicting = metadata + b"License: MIT\n"
        self.assertTrue(check_license.check_metadata(conflicting, "METADATA"))


if __name__ == "__main__":
    unittest.main()
