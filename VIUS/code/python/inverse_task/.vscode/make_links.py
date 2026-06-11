#!/usr/bin/env python3
import subprocess
import re
from pathlib import Path

USER = "EsaulVa"
REPO = "Projects"
BRANCH = "main"
# BASE_URL = f"https://raw.githubusercontent.com/{USER}/{REPO}/{BRANCH}"
BASE_URL="https://github.com/EsaulVa/Projects/tree/main/MatLab_inverse"

# Расширения, которые нужны (оставьте пустым, чтобы взять всё)
# ALLOWED_EXTS = {".py", ".md", ".txt", ".csv", ".json", ".html", ".yml", ".yaml",".m"}
# Расширения, которые нужны (оставьте пустым, чтобы взять всё)
ALLOWED_EXTS = {".py",".m"}

# Папки, которые игнорировать
SKIP_DIRS = {".git", ".vscode", "__pycache__", ".pytest_cache"}

result = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
files = result.stdout.strip().splitlines()

links = []
for f in files:
    p = Path(f)
    if any(part in SKIP_DIRS for part in p.parts):
        continue
    if ALLOWED_EXTS and p.suffix.lower() not in ALLOWED_EXTS:
        continue
    links.append(f"{BASE_URL}/{f}")

with open("raw_links.txt", "w", encoding="utf-8") as out:
    out.write("\n".join(links) + "\n")

print(f"Собрано {len(links)} ссылок → raw_links.txt")