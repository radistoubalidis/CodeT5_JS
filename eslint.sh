#!/bin/bash
if [ $# -eq 0 ]; then
    echo "provide batch index."
    exit 2
fi

npx eslint --format=json samples/*.js >> eslint_outputs/eslint_batch_$1.json

# npx eslint --format=json samples/*.js >> eslint_output.json