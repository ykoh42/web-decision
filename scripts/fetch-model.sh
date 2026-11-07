#!/usr/bin/env bash
# Fetch one pinned, third-party ONNX checkpoint for the comparison prototype.
set -euo pipefail
caller_dir=$PWD
cd "$(dirname "$0")/.."
revision=5a827091b3d58fe4fdfc02a5a5b2ce364047d678
base="https://huggingface.co/onnx-community/kev-0.6b-ONNX/resolve/$revision"
destination=models/kev-0.6b-onnx
with_q4=false
resume_name=''
resume_path=''
while (( $# )); do
  case "$1" in
    --with-q4)
      if [[ "$with_q4" == true ]]; then printf 'Duplicate --with-q4.\n' >&2; exit 1; fi
      with_q4=true
      shift
      ;;
    --resume)
      if [[ -n "$resume_name" || $# -lt 3 || -z "$2" || -z "$3" ]]; then
        printf 'Usage: %s [--with-q4] [--resume MODEL_FILE PARTIAL_FILE]\n' "$0" >&2
        exit 1
      fi
      resume_name=$2
      resume_path=$3
      if [[ "$resume_path" != /* ]]; then resume_path="$caller_dir/$resume_path"; fi
      shift 3
      ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 1 ;;
  esac
done
for path in models models/kev-0.6b-onnx models/kev-0.6b-onnx/onnx; do
  if [[ -L "$path" ]]; then
    printf 'Model output directory must not be a symlink: %s\n' "$path" >&2
    exit 1
  fi
done
if ! command -v python3 >/dev/null || ! python3 -c 'import os, sys; sys.exit(0 if os.link in os.supports_follow_symlinks else 1)'; then
  printf 'python3 with no-follow os.link support is required before preparing model files.\n' >&2
  exit 1
fi
graph_size=3338522
graph_sha='26e059ddd954efd9897772b098db10b9eeda9b873984dc1c73b32a9ce177c7c8'
link_exact() {
  python3 -c 'import os, sys; os.link(sys.argv[1], sys.argv[2], follow_symlinks=False)' "$1" "$2"
}
fetch_file() {
  local file="$1" expected="$2" expected_sha="$3" path="$destination/$1" actual actual_sha part backup_dir
  local replace=false
  if [[ -n "$resume_name" && "$resume_path" -ef "$path" ]]; then
    printf 'Cannot resume: partial is the same file as a selected destination: %s\n' "$path" >&2
    return 1
  fi
  if [[ -e "$path" || -L "$path" ]]; then
    if [[ ! -f "$path" || -L "$path" ]]; then
      printf 'Existing model path is not a regular file: %s\n' "$path" >&2
      return 1
    fi
    actual=$(wc -c < "$path" | tr -d ' ')
    if [[ "$actual" == "$expected" ]]; then
      actual_sha=$(shasum -a 256 "$path")
      actual_sha=${actual_sha%% *}
      if [[ "$actual_sha" == "$expected_sha" ]]; then
        if [[ "$file" == "$resume_name" ]]; then
          printf 'Cannot resume: verified model file already exists: %s\n' "$path" >&2
          return 1
        fi
        printf 'cached %s (%s bytes)\n' "$file" "$actual"
        return
      fi
    fi
    replace=true
    printf 'Existing model file differs; downloading a verified replacement before preserving it: %s\n' "$path" >&2
  fi
  part=$(mktemp "${path}.download.XXXXXXXX")
  local -a resume_args=()
  if [[ "$file" == "$resume_name" ]]; then
    if ! python3 - "$resume_path" "$part" "$expected" "$path" <<'PYRESUME'
import os
import shutil
import stat
import sys

source, destination, maximum, target = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
identity = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
try:
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ValueError('partial must be a nonempty regular file no larger than the pinned file')
        try:
            target_stat = os.lstat(target)
        except FileNotFoundError:
            target_stat = None
        if target_stat and (before.st_dev, before.st_ino) == (target_stat.st_dev, target_stat.st_ino):
            raise ValueError('partial is the same file as the destination')
        with open(destination, 'wb') as output:
            shutil.copyfileobj(stream, output, 1024 * 1024)
        if identity(before) != identity(os.fstat(stream.fileno())) or identity(before) != identity(os.lstat(source)):
            raise ValueError('partial changed while copying')
except (OSError, ValueError) as error:
    print(f'Cannot copy partial file: {error}', file=sys.stderr)
    sys.exit(1)
PYRESUME
    then
      printf 'Resume copy failed; temporary file retained at %s\n' "$part" >&2
      return 1
    fi
    actual=$(wc -c < "$part" | tr -d ' ')
    if [[ "$actual" != "$expected" ]]; then resume_args=(--continue-at -); fi
  fi
  printf 'fetching %s (%s bytes)\n' "$file" "$expected"
  if [[ "$file" == "$resume_name" && "$actual" == "$expected" ]]; then
    printf 'Checking complete retained file without a network request.\n'
  elif ! (
    ulimit -f "$(((expected + 511) / 512))" || exit 1
    curl --fail --location --silent --show-error --retry 2 --max-filesize "$expected" ${resume_args[@]+"${resume_args[@]}"} "$base/$file" -o "$part"
  )
  then
    printf 'Download failed; partial file retained at %s\n' "$part" >&2
    return 1
  fi
  actual=$(wc -c < "$part" | tr -d ' ')
  if [[ "$actual" != "$expected" ]]; then
    printf 'size mismatch for %s: expected %s, got %s; partial file retained at %s\n' "$file" "$expected" "$actual" "$part" >&2
    return 1
  fi
  actual_sha=$(shasum -a 256 "$part")
  actual_sha=${actual_sha%% *}
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    printf 'SHA-256 mismatch for %s; partial file retained at %s\n' "$file" "$part" >&2
    return 1
  fi
  if [[ "$replace" == true ]]; then
    if [[ -n "$resume_name" && "$resume_path" -ef "$path" ]]; then
      printf 'Cannot resume: partial is the same file as the destination: %s\n' "$path" >&2
      return 1
    fi
    if ! backup_dir=$(mktemp -d "${path}.backup.XXXXXXXX"); then
      printf 'Could not create backup directory; replacement retained at %s\n' "$part" >&2
      return 1
    fi
    if ! mv "$path" "$backup_dir/$(basename "$path")"; then
      printf 'Could not preserve current path; backup directory: %s; replacement retained at %s\n' "$backup_dir" "$part" >&2
      return 1
    fi
    printf 'Preserved existing model file at %s/%s\n' "$backup_dir" "$(basename "$path")" >&2
  fi
  if ! link_exact "$part" "$path"; then
    printf 'Model path could not be created without overwriting; replacement retained at %s\n' "$part" >&2
    if [[ "$replace" == true ]]; then printf 'Preserved previous path at %s/%s\n' "$backup_dir" "$(basename "$path")" >&2; fi
    return 1
  fi
  rm "$part"
}
files=(
  'config.json 2267 40ce1c875acf016ab7742ecdc45d3bad0fe8c51d3ead0ca2263ec5eb4e96f42d'
  'tokenizer.json 7031645 c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'
  'tokenizer_config.json 9678 3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5'
  "onnx/model_q4f16.onnx $graph_size $graph_sha"
  'onnx/model_q4f16.onnx_data 335357952 777335dbc052e2e02ec7591f6eefcfb29e2c28caf52075e492d226fc27e0824b'
)
if [[ "$with_q4" == true ]]; then
  files+=(
    'onnx/model_q4.onnx 1245327 db4657bb2f14f073cb4d289260dfc50af31adc1260a04cb2fe17c4f03da0881a'
    'onnx/model_q4.onnx_data 374822912 20668ed757f5de294c254cab111910fbc37fb11addca8189e3f58b3c695f90f1'
  )
fi
if [[ -n "$resume_name" ]]; then
  found=false
  selected_paths=()
  for spec in "${files[@]}"; do
    read -r file size sha <<< "$spec"
    selected_paths+=("$destination/$file")
    if [[ "$file" == "$resume_name" ]]; then found=true; resume_size=$size; fi
  done
  if [[ "$found" == false ]]; then
    printf 'Resume model file is not selected: %s\n' "$resume_name" >&2
    exit 1
  fi
  if ! python3 - "$resume_path" "$destination/$resume_name" "$resume_size" "${selected_paths[@]}" <<'PYCHECK'
import os
import stat
import sys

source, target, maximum = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        partial = os.fstat(fd)
        if not stat.S_ISREG(partial.st_mode) or not 0 < partial.st_size <= maximum:
            raise ValueError('partial must be a nonempty regular file no larger than the pinned file')
        for path in sys.argv[4:]:
            try:
                current = os.lstat(path)
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(current.st_mode):
                try:
                    linked = os.stat(path)
                except FileNotFoundError:
                    linked = None
            else:
                linked = current
            if linked and (partial.st_dev, partial.st_ino) == (linked.st_dev, linked.st_ino):
                raise ValueError(f'partial is the same file as a selected destination: {path}')
            if path == target and not stat.S_ISREG(current.st_mode):
                raise ValueError('destination is not a regular file')
    finally:
        os.close(fd)
except (OSError, ValueError) as error:
    print(f'Cannot copy partial file: {error}', file=sys.stderr)
    sys.exit(1)
PYCHECK
  then
    exit 1
  fi
fi
mkdir -p "$destination/onnx"
legacy="$destination/model_q4f16.onnx"
current="$destination/onnx/model_q4f16.onnx"
if [[ -L "$legacy" ]]; then
  printf 'Legacy model path must not be a symlink: %s\n' "$legacy" >&2
  exit 1
fi
if [[ -e "$legacy" && ! -f "$legacy" ]]; then
  printf 'Legacy model path is not a regular file: %s\n' "$legacy" >&2
  exit 1
fi
if [[ "$resume_name" != "onnx/model_q4f16.onnx" && !( -n "$resume_name" && "$resume_path" -ef "$legacy" ) && -f "$legacy" && ! -e "$current" && ! -L "$current" ]]; then
  legacy_size=$(wc -c < "$legacy" | tr -d ' ')
  legacy_sha=''
  if [[ "$legacy_size" == "$graph_size" ]]; then
    legacy_sha=$(shasum -a 256 "$legacy")
    legacy_sha=${legacy_sha%% *}
  fi
  if [[ "$legacy_size" == "$graph_size" && "$legacy_sha" == "$graph_sha" ]]; then
    if ! link_exact "$legacy" "$current"; then
      printf 'Could not link verified legacy file without overwriting: %s\n' "$current" >&2
      exit 1
    fi
  else
    printf 'Legacy model file differs; preserving it and using the normal download path: %s\n' "$legacy" >&2
  fi
fi
for spec in "${files[@]}"; do
  read -r file size sha <<< "$spec"
  fetch_file "$file" "$size" "$sha"
done
# Recheck all inputs after the last download, including previously cached files.
for spec in "${files[@]}"; do
  read -r file size sha <<< "$spec"
  path="$destination/$file"
  if [[ ! -f "$path" || -L "$path" ]] || [[ "$(wc -c < "$path" | tr -d ' ')" != "$size" ]]; then
    printf 'Final model file is missing or has the wrong size: %s\n' "$path" >&2
    exit 1
  fi
  actual_sha=$(shasum -a 256 "$path")
  if [[ "${actual_sha%% *}" != "$sha" ]]; then
    printf 'Final model file SHA-256 mismatch: %s\n' "$path" >&2
    exit 1
  fi
done
printf 'Pinned checkpoint ready at %s\n' "$destination"
