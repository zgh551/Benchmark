#!/bin/bash

test "$BASH_SOURCE" = "" && echo "This script can be sourced only from bash" && return

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

cd $SCRIPT_DIR

streamlit run sdnn.py
