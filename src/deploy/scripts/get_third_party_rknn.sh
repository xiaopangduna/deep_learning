#!/usr/bin/env bash
# filepath: scripts/get_third_party_rknn.sh
set -euo pipefail

# Fetch the `3rdparty` directory from the remote rknn_model_zoo repo.
# Usage:
#   ./scripts/get_third_party_rknn.sh [DEST] [BRANCH] [-f] [-m]
#     DEST    target directory (default: third_party)
#     BRANCH  git branch (default: main)
#     -f      force overwrite existing DEST (delete and replace)
#     -m      merge: merge fetched content into existing DEST (rsync, overwrite same files)
#
# Behavior:
#  - Download into a temporary directory first, prepare a TMP_DEST (no .git),
#    then either mv TMP_DEST -> DEST (atomic replace) for -f, rsync TMP_DEST -> DEST for -m,
#    or fail if DEST exists and neither -f nor -m specified.

REPO="https://github.com/airockchip/rknn_model_zoo.git"
SUBPATH="3rdparty"

DEST="third_party"
BRANCH="main"
FORCE=0
MERGE=0

usage() {
  cat <<EOF >&2
Usage: $0 [DEST] [BRANCH] [-f] [-m]
  DEST   target directory (default: third_party)
  BRANCH git branch (default: main)
  -f     overwrite existing DEST (delete and replace)
  -m     merge fetched content into existing DEST (rsync, overwrite same files)
EOF
  exit 1
}

# parse args
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
      if [[ -z "${_DEST_PROVIDED:-}" ]]; then
        DEST="$1"; _DEST_PROVIDED=1
      elif [[ -z "${_BRANCH_PROVIDED:-}" ]]; then
        BRANCH="$1"; _BRANCH_PROVIDED=1
      else
        echo "Ignoring extra positional argument: $1" >&2
      fi
      shift
      ;;
  esac
done

if [[ -d "$DEST" && $FORCE -ne 1 && $MERGE -ne 1 ]]; then
  echo "Destination '$DEST' already exists. Use -f to overwrite or -m to merge." >&2
  exit 1
fi

# temp workdir
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

_prepare_tmp_dest() {
  local SRC="$1"
  local TMP_DEST="$TMPDIR/tmp_deploy"
  mkdir -p "$TMP_DEST"
  if ! command -v rsync >/dev/null 2>&1; then
    echo "rsync required but not found" >&2
    exit 1
  fi
  # copy fetched content into TMP_DEST excluding any .git
  rsync -a --exclude='.git' "${SRC}/" "${TMP_DEST}/"
  # ensure no nested .git remains
  rm -rf "${TMP_DEST}/.git"
  echo "$TMP_DEST"
}

_deploy_from_tmp() {
  local TMP_DEST="$1"
  if [[ -d "$DEST" ]]; then
    if [[ $MERGE -eq 1 ]]; then
      echo "Merging TMP content into existing '$DEST' (rsync, will overwrite same files, keep extras)..."
      rsync -a "${TMP_DEST}/" "${DEST}/"
    elif [[ $FORCE -eq 1 ]]; then
      echo "Replacing existing '$DEST' with TMP content..."
      rm -rf "$DEST"
      mv "$TMP_DEST" "$DEST"
    else
      echo "Destination exists and neither -f nor -m specified; aborting." >&2
      exit 1
    fi
  else
    echo "Installing TMP content to '$DEST'..."
    mv "$TMP_DEST" "$DEST"
  fi
  echo "Deploy complete: ${SUBPATH} -> ${DEST}"
}

# fetch via git sparse-checkout (preferred)
if command -v git >/dev/null 2>&1; then
  echo "Cloning sparse ${SUBPATH} from ${REPO} (branch=${BRANCH}) into tmp..."
  git clone --depth 1 --filter=blob:none --sparse --branch "$BRANCH" "$REPO" "$TMPDIR"
  pushd "$TMPDIR" >/dev/null
  git sparse-checkout init --cone >/dev/null
  git sparse-checkout set "$SUBPATH" >/dev/null
  popd >/dev/null

  SRC="$TMPDIR/$SUBPATH"
  if [[ ! -d "$SRC" ]]; then
    echo "Sparse-checkout failed to produce expected path: $SRC" >&2
    exit 1
  fi

  TMP_DEST=$(_prepare_tmp_dest "$SRC")
  _deploy_from_tmp "$TMP_DEST"
  exit 0
fi

# fallback: download zip and extract
echo "git not available; downloading ZIP fallback..."
ZIPURL="https://github.com/airockchip/rknn_model_zoo/archive/refs/heads/${BRANCH}.zip"
if ! command -v curl >/dev/null 2>&1 || ! command -v unzip >/dev/null 2>&1; then
  echo "curl and unzip required for ZIP fallback" >&2
  exit 1
fi

curl -L -o "$TMPDIR/repo.zip" "$ZIPURL"
unzip -q "$TMPDIR/repo.zip" -d "$TMPDIR"
TOPDIR="$(find "$TMPDIR" -maxdepth 1 -type d -name 'rknn_model_zoo-*' -print -quit)"
if [[ -z "$TOPDIR" ]]; then
  echo "Failed to locate extracted repo top directory" >&2
  exit 1
fi

SRC="$TOPDIR/$SUBPATH"
if [[ ! -d "$SRC" ]]; then
  echo "Subpath $SUBPATH not found in archive" >&2
  exit 1
fi

TMP_DEST=$(_prepare_tmp_dest "$SRC")
_deploy_from_tmp "$TMP_DEST"