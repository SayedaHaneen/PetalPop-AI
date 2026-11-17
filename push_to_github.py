"""
Direct GitHub Uploader for PetalPop AI.
Uses GitHub REST API (Git Trees & Blobs) to push all project files directly
to https://github.com/SayedaHaneen/PetalPop-AI without needing git.exe installed.

Usage:
    python push_to_github.py <YOUR_GITHUB_TOKEN>
"""

import os
import sys
import base64
import requests

REPO_NAME = "PetalPop-AI"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Exclude large binary datasets and cache
IGNORE_PATTERNS = [
    "data/raw",
    "data/processed",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".DS_Store",
    "venv",
    ".venv"
]

def should_ignore(rel_path: str) -> bool:
    rel_path_fwd = rel_path.replace("\\", "/")
    for pattern in IGNORE_PATTERNS:
        if rel_path_fwd == pattern or rel_path_fwd.startswith(pattern + "/"):
            return True
        if "/__pycache__" in rel_path_fwd or rel_path_fwd.endswith(".pyc"):
            return True
    return False

def push_project_to_github(token: str):
    headers = {
        "Authorization": f"token {token.strip()}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "PetalPopAI-Uploader"
    }

    # 1. Verify user
    print("Verifying GitHub credentials...", flush=True)
    r = requests.get("https://api.github.com/user", headers=headers)
    if r.status_code != 200:
        print(f"Authentication failed ({r.status_code}): {r.text}", flush=True)
        print("\nPlease ensure your token has 'repo' scope and has not been revoked.", flush=True)
        return False
    username = r.json()["login"]
    print(f"Authenticated as: {username}", flush=True)

    # 2. Check / Create Repository
    repo_url = f"https://api.github.com/repos/{username}/{REPO_NAME}"
    r_repo = requests.get(repo_url, headers=headers)
    if r_repo.status_code == 404:
        print(f"Creating repository '{REPO_NAME}' on GitHub...", flush=True)
        create_payload = {
            "name": REPO_NAME,
            "description": "PetalPop AI 🌸: Intelligent Flower Recognition and Care Assistant using MobileNetV2 and Streamlit",
            "private": False,
            "auto_init": True
        }
        r_create = requests.post("https://api.github.com/user/repos", headers=headers, json=create_payload)
        if r_create.status_code not in (200, 201):
            print(f"Failed to create repo: {r_create.text}", flush=True)
            return False
        print(f"Repository created successfully at {r_create.json()['html_url']}", flush=True)
    elif r_repo.status_code == 200:
        print(f"Repository exists at: {r_repo.json()['html_url']}", flush=True)

    # 3. Collect files to upload
    files_to_upload = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, PROJECT_ROOT)
            if not should_ignore(rel_path):
                # Don't upload individual files larger than 50MB via tree API
                if os.path.getsize(full_path) < 50 * 1024 * 1024:
                    files_to_upload.append((rel_path, full_path))

    print(f"Found {len(files_to_upload)} files to synchronize with GitHub.", flush=True)

    # 4. Create blobs
    tree_items = []
    for idx, (rel_path, full_path) in enumerate(files_to_upload, start=1):
        rel_path_fwd = rel_path.replace("\\", "/")
        try:
            with open(full_path, "rb") as fp:
                content_bytes = fp.read()
            encoded_content = base64.b64encode(content_bytes).decode("utf-8")
            blob_resp = requests.post(
                f"https://api.github.com/repos/{username}/{REPO_NAME}/git/blobs",
                headers=headers,
                json={"content": encoded_content, "encoding": "base64"}
            )
            if blob_resp.status_code in (200, 201):
                blob_sha = blob_resp.json()["sha"]
                tree_items.append({
                    "path": rel_path_fwd,
                    "mode": "100644",
                    "type": "blob",
                    "sha": blob_sha
                })
                print(f"[{idx}/{len(files_to_upload)}] Uploaded {rel_path_fwd}", flush=True)
            else:
                print(f"Failed blob for {rel_path_fwd}: {blob_resp.text}", flush=True)
        except Exception as e:
            print(f"Error processing {rel_path}: {e}", flush=True)

    # 5. Create Tree
    print("Creating Git tree...", flush=True)
    tree_resp = requests.post(
        f"https://api.github.com/repos/{username}/{REPO_NAME}/git/trees",
        headers=headers,
        json={"tree": tree_items}
    )
    if tree_resp.status_code not in (200, 201):
        print(f"Failed to create tree: {tree_resp.text}", flush=True)
        return False
    tree_sha = tree_resp.json()["sha"]

    # 6. Get Parent Commit (if any)
    ref_resp = requests.get(
        f"https://api.github.com/repos/{username}/{REPO_NAME}/git/refs/heads/main",
        headers=headers
    )
    parent_commits = []
    if ref_resp.status_code == 200:
        parent_commits.append(ref_resp.json()["object"]["sha"])

    # 7. Create Commit
    print("Creating commit...", flush=True)
    commit_payload = {
        "message": "🌸 Release: PetalPop AI - Intelligent Flower Recognition & Care Assistant",
        "tree": tree_sha,
        "parents": parent_commits
    }
    commit_resp = requests.post(
        f"https://api.github.com/repos/{username}/{REPO_NAME}/git/commits",
        headers=headers,
        json=commit_payload
    )
    if commit_resp.status_code not in (200, 201):
        print(f"Failed to create commit: {commit_resp.text}", flush=True)
        return False
    commit_sha = commit_resp.json()["sha"]

    # 8. Update Reference (branch main)
    print("Updating main branch...", flush=True)
    if parent_commits:
        update_resp = requests.patch(
            f"https://api.github.com/repos/{username}/{REPO_NAME}/git/refs/heads/main",
            headers=headers,
            json={"sha": commit_sha, "force": True}
        )
    else:
        update_resp = requests.post(
            f"https://api.github.com/repos/{username}/{REPO_NAME}/git/refs",
            headers=headers,
            json={"ref": "refs/heads/main", "sha": commit_sha}
        )

    if update_resp.status_code in (200, 201):
        live_link = f"https://github.com/{username}/{REPO_NAME}"
        print("\n" + "="*60, flush=True)
        print(f"🎉 SUCCESS! Project pushed to GitHub!", flush=True)
        print(f"🌐 LIVE WORKING GITHUB LINK: {live_link}", flush=True)
        print("="*60 + "\n", flush=True)
        return True
    else:
        print(f"Failed to update ref: {update_resp.text}", flush=True)
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        token = sys.argv[1]
    else:
        token = os.environ.get("GITHUB_TOKEN", "")
    
    if not token:
        print("Usage: python push_to_github.py <GITHUB_TOKEN>")
        sys.exit(1)

    push_project_to_github(token)
