# SPDX-License-Identifier: Apache-2.0
"""Check source license notices and the licenses shipped in PDHCG packages."""

import argparse
from email.parser import BytesParser
import hashlib
from pathlib import Path
import re
import subprocess
import tarfile
from zipfile import BadZipFile, ZipFile

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


LICENSE_ID = "Apache-2.0"
# Reviewed Apache terms and existing PDHCG/cuPDLPx attribution. Changing this
# digest requires reviewing the license change; never regenerate it in CI.
LICENSE_SHA256 = "326679c4830fda99c8c9e06a19329530b274507e06e0bb8ea8cd1fb42f8b4d3a"
SOURCE_SUFFIXES = {".c", ".h", ".cu", ".cuh", ".cpp", ".hpp", ".py", ".pyi", ".cmake", ".js"}
SOURCE_DIRS = (
    "src", "include", "internal", "distributed", "python", "pdhcg", "python_bindings",
    "cmake", "test", "tests", "docs", ".github",
)


def is_source(path):
    return (
        path.suffix in SOURCE_SUFFIXES
        or path.name == "CMakeLists.txt"
        or path.name.endswith(".py.in")
    )


def source_files(root):
    """Use Git's file selection, or explicit source directories in an export."""
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode == 0 and Path(result.stdout.strip()).resolve() == root.resolve():
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            capture_output=True, text=True, check=True,
        )
        paths = {root / name for name in result.stdout.split("\0") if name}
    else:
        paths = set(root.iterdir())
        for directory in SOURCE_DIRS:
            paths.update((root / directory).rglob("*"))
    return sorted(path for path in paths if path.is_file() and is_source(path))


def has_apache_notice(text, path):
    hash_comments = (
        path.suffix in {".py", ".pyi", ".cmake"}
        or path.name == "CMakeLists.txt"
        or path.name.endswith(".py.in")
    )
    comments = []
    if text.startswith("#!"):
        text = text.partition("\n")[2]
    while text:
        text = text.lstrip()
        if text.startswith("/*"):
            comment, _, text = text[2:].partition("*/")
        elif text.startswith("//") or (hash_comments and text.startswith("#")):
            comment, _, text = text.partition("\n")
        else:
            break
        comments.append(comment)
    header = "\n".join(comments)
    identifiers = re.findall(r"SPDX-License-Identifier:\s*([^\r\n]+)", header)
    if identifiers:
        return all(value.strip() == LICENSE_ID for value in identifiers)
    return bool(re.search(r"Licensed under the Apache License,\s*Version 2\.0\b", header))


def check_repository(root):
    errors = []
    license_path = root / "LICENSE"
    if not license_path.is_file():
        errors.append("LICENSE: missing Apache license text")
    elif hashlib.sha256(license_path.read_bytes()).hexdigest() != LICENSE_SHA256:
        errors.append("LICENSE: reviewed license terms or attribution changed")

    try:
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        identifier = project.get("license")
        if isinstance(identifier, dict):
            identifier = identifier.get("text")
        if identifier != LICENSE_ID:
            errors.append("pyproject.toml: project.license must identify Apache-2.0")
    except (OSError, ValueError, KeyError) as error:
        errors.append(f"pyproject.toml: cannot read project metadata: {error}")

    for path in source_files(root):
        if not has_apache_notice(path.read_text(encoding="utf-8"), path):
            errors.append(f"{path.relative_to(root)}: missing or unsupported Apache-2.0 notice")
    return errors


def check_metadata(data, location):
    metadata = BytesParser().parsebytes(data)
    identifiers = metadata.get_all("License-Expression", []) + metadata.get_all("License", [])
    if not identifiers or any(value.strip() != LICENSE_ID for value in identifiers):
        return [f"{location}: package metadata must identify Apache-2.0"]
    if metadata.get("Name", "").lower().replace("_", "-") != "pdhcg":
        return [f"{location}: expected PDHCG package metadata"]
    return []


def check_archive_files(names, read, artifact, license_text, wheel):
    metadata_names = [
        name for name in names
        if len(Path(name).parts) == 2 and name.endswith(".dist-info/METADATA" if wheel else "/PKG-INFO")
    ]
    if len(metadata_names) != 1:
        return [f"{artifact}: expected exactly one package metadata file"]
    metadata_name = metadata_names[0]
    errors = check_metadata(read(metadata_name), f"{artifact}:{metadata_name}")
    prefix = metadata_name.rsplit("/", 1)[0]
    candidates = [f"{prefix}/LICENSE"]
    if wheel:
        candidates.append(f"{prefix}/licenses/LICENSE")
    licenses = [name for name in candidates if name in names]
    if not licenses:
        errors.append(f"{artifact}: missing packaged LICENSE")
    for name in licenses:
        if read(name) != license_text:
            errors.append(f"{artifact}:{name}: differs from repository LICENSE")
    return errors


def check_artifact(artifact, license_text):
    try:
        if artifact.suffix == ".whl":
            with ZipFile(artifact) as archive:
                return check_archive_files(archive.namelist(), archive.read, artifact, license_text, True)
        if artifact.name.endswith(".tar.gz"):
            with tarfile.open(artifact, "r:gz") as archive:
                names = [member.name for member in archive.getmembers() if member.isfile()]

                def read(name):
                    with archive.extractfile(name) as stream:
                        return stream.read()

                return check_archive_files(names, read, artifact, license_text, False)
        return [f"{artifact}: expected a .whl or .tar.gz package"]
    except (OSError, ValueError, KeyError, BadZipFile, tarfile.TarError) as error:
        return [f"{artifact}: cannot inspect package: {error}"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifacts", "--wheel", type=Path, nargs="+", default=[])
    args = parser.parse_args()
    root = args.root.resolve()
    errors = check_repository(root)
    if (root / "LICENSE").is_file():
        for artifact in args.artifacts:
            errors.extend(check_artifact(artifact, (root / "LICENSE").read_bytes()))
    if errors:
        parser.exit(1, "\n".join(errors) + "\n")
    print(f"License checks passed ({len(source_files(root))} source files, {len(args.artifacts)} packages).")


if __name__ == "__main__":
    main()
