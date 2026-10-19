#!/bin/bash
# Installed to ~/work/kai-main/up.sh. Upstream's ./glm53 wrapper adds every overlay.
cd "$(dirname "$0")" && exec ./glm53 "$@"
