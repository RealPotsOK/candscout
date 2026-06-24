#!/usr/bin/env bash
set -euo pipefail
cmd="${1:?usage: github.sh check|init|commit|create-private|push|publish}"
MAKE_BIN="${MAKE:-make}"
GITHUB_REPO="${GITHUB_REPO:-candscout}"
COMMIT_MSG="${COMMIT_MSG:-organize candscout project}"

check() {
  command -v git >/dev/null 2>&1 || { echo "Missing git. Install git, then rerun make github-publish."; exit 1; }
  command -v gh >/dev/null 2>&1 || { echo "Missing GitHub CLI 'gh'. Install gh, run 'gh auth login', then rerun make github-publish."; exit 1; }
  gh auth status >/dev/null 2>&1 || { echo "GitHub CLI is not authenticated. Run 'gh auth login', then rerun make github-publish."; exit 1; }
  echo "GitHub CLI is installed and authenticated."
}

init_repo() {
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Already inside a git repository."
  else
    git init -b main
  fi
  git branch -M main
}

case "$cmd" in
  check)
    check
    ;;
  init)
    init_repo
    ;;
  commit)
    init_repo
    git var GIT_AUTHOR_IDENT >/dev/null 2>&1 || { echo "Missing git author identity. Run: git config --global user.name 'Your Name' && git config --global user.email 'you@example.com'"; exit 1; }
    git add .gitignore .github/workflows/smoke.yml README.md docs Makefile requirements.txt env src scripts live_sim
    if git diff --cached --quiet; then
      echo "No source/config/docs changes staged for commit."
    else
      git commit -m "$COMMIT_MSG"
    fi
    ;;
  create-private)
    check
    init_repo
    if git remote get-url origin >/dev/null 2>&1; then
      echo "origin already exists: $(git remote get-url origin)"
    else
      gh repo create "$GITHUB_REPO" --private --source=. --remote=origin
    fi
    ;;
  push)
    check
    init_repo
    git remote get-url origin >/dev/null 2>&1 || { echo "Missing origin remote. Run make github-create-private first."; exit 1; }
    git push -u origin main
    ;;
  publish)
    "$MAKE_BIN" github-check
    "$MAKE_BIN" github-init
    "$MAKE_BIN" github-commit
    "$MAKE_BIN" github-create-private
    "$MAKE_BIN" github-push
    ;;
  *)
    echo "Unknown github command: $cmd" >&2
    exit 2
    ;;
esac
