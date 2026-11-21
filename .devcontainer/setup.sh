#!/usr/bin/env bash
# Run once when the codespace is created: the system libraries Qt needs, fonts, pymetrix with every extra and the
# first-party plugins in packages/ (finance, and the research lab book), and JDemetra+'s libraries for TRAMO-SEATS
# (Java 21 comes from the Java feature in devcontainer.json).
set -euo pipefail

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    libgl1 libegl1 libdbus-1-3 libfontconfig1 libxkbcommon-x11-0 \
    libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
    libxcb-shape0 libxcb-xfixes0 libxcb-xinerama0 libxcb-xkb1 \
    x11-utils fonts-dejavu fonts-liberation
sudo rm -rf /var/lib/apt/lists/*

python -m pip install --upgrade pip
python -m pip install -e ".[all]" -e packages/pymetrix-finance -e packages/pymetrix-labbook -e "packages/pymetrix-seasonal[jdemetra]" pytest

# JDemetra+'s jars, about 7 MB from Maven Central, each checked against its recorded SHA-1. A failure here leaves the
# rest working: TRAMO-SEATS then says what is missing, and the same line run in the terminal fetches them.
python -c "import pymetrix_seasonal.jdemetra as j; print('JDemetra+ libraries in', j.install())" \
    || echo "TRAMO-SEATS: JDemetra+'s libraries were not fetched; run: python -c 'import pymetrix_seasonal.jdemetra as j; j.install()'"

echo "PyMetrix is installed, with the finance, lab book and seasonal plugins (X-13 and TRAMO-SEATS). Open the 'PyMetrix desktop' port (6080) to see the app; the password is 'vscode'."
