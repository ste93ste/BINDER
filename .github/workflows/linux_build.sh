#!/bin/bash -e
# Build a BINDER CPU wheel for the given Python and install it (Linux).
PYTHON_PATH=$1

WHEELDIR=$(mktemp -d)
$PYTHON_PATH -m pip wheel . -w "$WHEELDIR" --no-deps \
    --config-settings=cmake.define.BINDER_FORCE_CPU=ON

# Install the built wheel by path
$PYTHON_PATH -m pip install "$WHEELDIR"/*.whl

# Smoke-test from a neutral dir so we import the installed package, not ./BINDER
(cd /tmp && $PYTHON_PATH -c 'import BINDER; print(BINDER.__file__, BINDER._backend)')

mkdir -p ./dist
cp "$WHEELDIR"/*.whl ./dist/
