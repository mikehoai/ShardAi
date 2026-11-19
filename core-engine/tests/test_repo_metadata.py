"""Unit tests verifying client-only README boundaries and repository homepage metadata."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
CANONICAL_HOMEPAGE: Final[str] = "https://shardai.online"
CANONICAL_REPO_URL: Final[str] = "https://github.com/mikehoai/ShardAi"
FORBIDDEN_CLOUD_TERMS: Final[tuple[str, ...]] = (
    "cloudflare",
    "hono",
    "wrangler",
    "shardai-cloud",
    "cloudflare kv",
    "cloudflare d1",
    "cloudflare workers",
    "cloudflare edge",
    "admin_secret_key",
)


def test_readme_features_canonical_homepage_and_client_only_scope() -> None:
    """README.md must link to https://shardai.online and document shardai-node client layers."""
    readme_path: Path = REPO_ROOT / "README.md"
    assert readme_path.is_file(), "README.md must exist at repository root"

    content: str = readme_path.read_text(encoding="utf-8")

    # Canonical homepage and repository links
    assert CANONICAL_HOMEPAGE in content
    assert CANONICAL_REPO_URL in content
    assert "sharidai.online" not in content.lower()

    # Client-only architectural components and workflows
    assert "shardai-node" in content
    assert "/v1/chat/completions" in content
    assert "src-ui/" in content
    assert "src-tauri/" in content
    assert "core-engine/" in content
    assert "shard_engine/" in content
    assert "SHARDAI_TRACKER_URL" in content
    assert "SHARDAI_OFFLINE_MODE" in content
    assert "docs/BUILD.md" in content

    # Stale paths and package manager references must not remain
    assert "core-engine/api/" not in content
    assert "│   ├── api/" not in content
    assert re.search(r"\bpnpm\b", content) is None


def test_readme_enforces_strict_public_private_boundary() -> None:
    """README.md must not leak any private shardai-cloud infrastructure or stack details."""
    readme_path: Path = REPO_ROOT / "README.md"
    content_lower: str = readme_path.read_text(encoding="utf-8").lower()

    for term in FORBIDDEN_CLOUD_TERMS:
        assert term not in content_lower, f"Forbidden private cloud term '{term}' found in README.md"

    assert re.search(r"\bd1\b", content_lower) is None


def test_npm_package_manifests_declare_canonical_homepage() -> None:
    """Root package.json and src-ui/package.json must declare https://shardai.online."""
    manifest_paths: tuple[Path, ...] = (
        REPO_ROOT / "package.json",
        REPO_ROOT / "src-ui" / "package.json",
    )

    for manifest_path in manifest_paths:
        assert manifest_path.is_file(), f"Missing manifest: {manifest_path}"
        raw_text: str = manifest_path.read_text(encoding="utf-8")
        data: dict[str, object] = json.loads(raw_text)

        assert data.get("homepage") == CANONICAL_HOMEPAGE
        assert data.get("license") == "MIT"

        repo_field: object = data.get("repository")
        assert isinstance(repo_field, dict)
        assert repo_field.get("url") == f"{CANONICAL_REPO_URL}.git"

        bugs_field: object = data.get("bugs")
        assert isinstance(bugs_field, dict)
        assert bugs_field.get("url") == f"{CANONICAL_REPO_URL}/issues"

        lower_text: str = raw_text.lower()
        assert "sharidai.online" not in lower_text
        for term in FORBIDDEN_CLOUD_TERMS:
            assert term not in lower_text


def test_tauri_and_cargo_manifests_declare_canonical_homepage() -> None:
    """src-tauri/Cargo.toml and tauri.conf.json must include https://shardai.online."""
    cargo_path: Path = REPO_ROOT / "src-tauri" / "Cargo.toml"
    cargo_data: dict[str, object] = tomllib.loads(cargo_path.read_text(encoding="utf-8"))
    package_section: object = cargo_data.get("package")
    assert isinstance(package_section, dict)
    assert package_section.get("homepage") == CANONICAL_HOMEPAGE
    assert package_section.get("repository") == CANONICAL_REPO_URL
    assert package_section.get("license") == "MIT"

    tauri_conf_path: Path = REPO_ROOT / "src-tauri" / "tauri.conf.json"
    tauri_data: dict[str, object] = json.loads(tauri_conf_path.read_text(encoding="utf-8"))
    bundle_section: object = tauri_data.get("bundle")
    assert isinstance(bundle_section, dict)
    assert bundle_section.get("homepage") == CANONICAL_HOMEPAGE


def test_pyproject_and_github_repository_metadata_declare_canonical_homepage() -> None:
    """core-engine/pyproject.toml and .github/repository-metadata.json must match canonical URLs."""
    pyproject_path: Path = REPO_ROOT / "core-engine" / "pyproject.toml"
    pyproject_data: dict[str, object] = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project_section: object = pyproject_data.get("project")
    assert isinstance(project_section, dict)

    urls_section: object = project_section.get("urls")
    assert isinstance(urls_section, dict)
    assert urls_section.get("Homepage") == CANONICAL_HOMEPAGE
    assert urls_section.get("Website") == CANONICAL_HOMEPAGE
    assert urls_section.get("Repository") == CANONICAL_REPO_URL
    assert urls_section.get("Issues") == f"{CANONICAL_REPO_URL}/issues"

    github_meta_path: Path = REPO_ROOT / ".github" / "repository-metadata.json"
    assert github_meta_path.is_file(), ".github/repository-metadata.json must exist"
    github_raw: str = github_meta_path.read_text(encoding="utf-8")
    github_meta: dict[str, object] = json.loads(github_raw)

    assert github_meta.get("homepage") == CANONICAL_HOMEPAGE
    assert github_meta.get("repository") == CANONICAL_REPO_URL
    assert github_meta.get("visibility") == "public"
    assert github_meta.get("component") == "shardai-node"
    assert "sharidai.online" not in github_raw.lower()

    for term in FORBIDDEN_CLOUD_TERMS:
        assert term not in github_raw.lower()
