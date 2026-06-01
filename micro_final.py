# -*- coding: utf-8 -*-
"""Backward-compatible entry point for PlantGuard AI.

The original 934-line Colab-derived prototype has been refactored into the
``plantguard`` package (see ROADMAP.md [6.1] "934-line monolith"). This shim
keeps the familiar ``python micro_final.py`` command working.

The original single-file version is preserved in git history (commit 33e4c7a)
if you ever need to refer back to it.
"""

from plantguard.app import main

if __name__ == "__main__":
    main()
