#!/usr/bin/env bash
set -euo pipefail

lab_dir="${1:-toy/dvc/workspace}"
remote_dir="${2:-toy/dvc/remote}"

mkdir -p "$lab_dir" "$remote_dir"
cd "$lab_dir"
git init
dvc init
git config user.name "${GIT_AUTHOR_NAME:-VisionGuard Student}"
git config user.email "${GIT_AUTHOR_EMAIL:-student@example.com}"

printf 'sample_id,value\n1,10\n2,20\n' > observations.csv
dvc add observations.csv
git add .dvc .dvcignore observations.csv.dvc .gitignore
git commit -m "Track first toy dataset version"
first_commit="$(git rev-parse HEAD)"

printf '3,30\n' >> observations.csv
dvc add observations.csv
git add observations.csv.dvc
git commit -m "Track second toy dataset version"

dvc remote add --local toy "$(cd "../remote" && pwd)"
dvc push -r toy
rm observations.csv
dvc pull -r toy

printf 'First data commit: %s\nCurrent data commit: %s\n' "$first_commit" "$(git rev-parse HEAD)"
