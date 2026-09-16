#!/bin/bash -e
# Build a BINDER CPU wheel for the given Python and install it (macOS).
PYTHON_PATH=$1

WHEELDIR=$(mktemp -d)
$PYTHON_PATH -m pip wheel . -w "$WHEELDIR" --no-deps \
    --config-settings=cmake.define.BINDEREG_FORCE_CPU=ON

$PYTHON_PATH -m pip install "$WHEELDIR"/*.whl
(cd /tmp && $PYTHON_PATH -c 'import BINDER; print(BINDER.__file__, BINDER._backend)')

mkdir -p ./dist
cp "$WHEELDIR"/*.whl ./dist/
