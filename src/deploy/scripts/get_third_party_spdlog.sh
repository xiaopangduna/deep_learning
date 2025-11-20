#!/usr/bin/env bash
set -euo pipefail

# Fetch spdlog into third_party/spdlog safely:
# Usage: ./scripts/get_third_party_spdlog.sh [DEST] [REF] [-f] [-m]
#  DEST default: third_party/spdlog
#  REF  default: v1.x
#  -m   merge (default if DEST exists): rsync into DEST (keep extra local files)
#  -f   force replace DEST with fetched content (DEST removed then moved)
# Examples:
#  ./scripts/get_third_party_spdlog.sh
#  ./scripts/get_third_party_spdlog.sh -m
#  ./scripts/get_third_party_spdlog.sh -f
#  ./scripts/get_third_party_spdlog.sh third_party/spdlog v1.11.0 -f

REPO="https://github.com/gabime/spdlog.git"
DEST="third_party/spdlog"
REF="v1.x"
FORCE=0
MERGE=0

usage() {
  cat <<EOF >&2
Usage: $0 [DEST] [REF] [-f] [-m]
  -f  force replace DEST
  -m  merge into DEST (rsync)
EOF
  exit 1
}

# parse args robustly
_pos=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -f) FORCE=1; shift ;;
    -m|--merge) MERGE=1; shift ;;
    -h|--help) usage ;;
    --) shift; break ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      ;;
    *)
      if [[ $_pos -eq 0 ]]; then DEST="$1"
      elif [[ $_pos -eq 1 ]]; then REF="$1"
      else echo "Ignoring extra arg: $1" >&2
      fi
      _pos=$(( _pos + 1 ))
      shift
      ;;
  esac
done

if [[ -d "$DEST" && $FORCE -ne 1 && $MERGE -ne 1 ]]; then
  echo "Destination '$DEST' exists. Use -m to merge or -f to force replace." >&2
  exit 1
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

_safe_mkdir_parent() { mkdir -p "$(dirname -- "$1")"; }

_clone_to_tmp() {
  if command -v git >/dev/null 2>&1; then
    echo "Cloning ${REPO} (${REF}) into temporary dir..."
    if git clone --depth 1 --branch "${REF}" "${REPO}" "$TMPDIR" 2>/dev/null; then
      : # ok
    else
      echo "Specified ref '${REF}' not found or clone with ref failed, falling back to default branch..."
      git clone --depth 1 "${REPO}" "$TMPDIR"
    fi
    echo "Clone complete."
  else
    # fallback ZIP
    if ! command -v curl >/dev/null 2>&1 || ! command -v unzip >/dev/null 2>&1; then
      echo "git or (curl+unzip) required" >&2
      exit 1
    fi
    ZIPURL="https://github.com/gabime/spdlog/archive/refs/heads/${REF}.zip"
    echo "Downloading ${ZIPURL} ..."
    curl -L -o "$TMPDIR/spdlog.zip" "$ZIPURL" || true
    if [[ ! -f "$TMPDIR/spdlog.zip" ]]; then
      ZIPURL="https://github.com/gabime/spdlog/archive/refs/heads/main.zip"
      curl -L -o "$TMPDIR/spdlog.zip" "$ZIPURL"
    fi
    unzip -q "$TMPDIR/spdlog.zip" -d "$TMPDIR"
    TOPDIR="$(find "$TMPDIR" -maxdepth 1 -type d -name 'spdlog-*' -print -quit)"
    if [[ -z "$TOPDIR" ]]; then
      echo "Failed to extract spdlog archive" >&2
      exit 1
    fi
    TMPDIR="$TOPDIR"
  fi
}

_copy_exclude_git() {
  local SRC="$1"
  _safe_mkdir_parent "$DEST"
  if command -v rsync >/dev/null 2>&1; then
    # ensure DEST exists if merging; for force we'll recreate
    if [[ $FORCE -eq 1 ]]; then
      rm -rf "$DEST"
      mkdir -p "$DEST"
    fi
    rsync -a --exclude='.git' --delete "${SRC}/" "${DEST}/"
  else
    # fallback cp
    if [[ $FORCE -eq 1 ]]; then rm -rf "$DEST"; fi
    mkdir -p "$DEST"
    cp -a "${SRC}/." "$DEST/"
    rm -rf "${DEST}/.git" || true
  fi
}

# perform
_clone_to_tmp

# TMPDIR now points to the checked-out repo root or extracted dir
# copy content into a safe temporary deploy dir first to ensure no nested .git ends up in DEST
TMP_DEPLOY="$(mktemp -d)"
trap 'rm -rf "$TMP_DEPLOY" "$TMPDIR"' EXIT
if command -v rsync >/dev/null 2>&1; then
  rsync -a --exclude='.git' "${TMPDIR}/" "${TMP_DEPLOY}/"
else
  cp -a "${TMPDIR}/." "${TMP_DEPLOY}/"
  rm -rf "${TMP_DEPLOY}/.git" || true
fi

# deploy from TMP_DEPLOY to DEST according to flags
if [[ -d "$DEST" ]]; then
  if [[ $MERGE -eq 1 ]]; then
    echo "Merging into existing ${DEST}..."
    rsync -a --exclude='.git' --delete "${TMP_DEPLOY}/" "${DEST}/"
  elif [[ $FORCE -eq 1 ]]; then
    echo "Force replacing ${DEST}..."
    rm -rf "$DEST"
    mv "$TMP_DEPLOY" "$DEST"
  else
    echo "Destination exists and neither -m nor -f specified; aborting." >&2
    exit 1
  fi
else
  echo "Installing fetched spdlog to ${DEST}..."
  mv "$TMP_DEPLOY" "$DEST"
fi

echo "Done: spdlog ->