#!/usr/bin/env bash
set -euo pipefail
pip-compile --strip-extras -q -o requirements-dev.txt requirements-dev.in
pip-compile --strip-extras -q -o requirements.txt requirements.in
pip-sync requirements-dev.txt
