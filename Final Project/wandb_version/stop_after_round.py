#!/usr/bin/env python3
"""Create .stop_after_round in the project root. The coordinator will exit after the current round completes."""
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
stop_file = os.path.join(project_root, ".stop_after_round")
with open(stop_file, "w") as f:
    f.write("")
print(f"Created {stop_file}. Coordinator will stop after the current round completes.")
