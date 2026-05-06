"""Extract readable conversation from Claude Code transcript file."""
import json

TRANSCRIPT = r"C:\Users\ChengYuxuan\OneDrive\ClaudeConfig\projects\D--ArtiIntComVis-MOSAIC\426427a3-fe5c-410f-a473-605e207a24e2.jsonl"

with open(TRANSCRIPT, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}\n")

for i, line in enumerate(lines, 1):
    try:
        data = json.loads(line.strip())
    except json.JSONDecodeError:
        continue

    msg_type = data.get("type", "")
    timestamp = data.get("timestamp", "")[:19].replace("T", " ")

    if msg_type == "user" and "message" in data:
        for c in data["message"].get("content", []):
            if not isinstance(c, dict):
                continue
            if c.get("type") == "text":
                text = c["text"]
                print(f"{'='*70}")
                print(f"[L{i}] [{timestamp}] YOU:")
                print(text)
                print()

    elif msg_type == "assistant" and "message" in data:
        for c in data["message"].get("content", []):
            if not isinstance(c, dict):
                continue
            if c.get("type") == "text":
                text = c["text"]
                print(f"{'='*70}")
                print(f"[L{i}] [{timestamp}] CLAUDE:")
                print(text)
                print()

    elif msg_type == "last-prompt":
        print(f"{'='*70}")
        print(f"[L{i}] --- Last prompt: {data.get('lastPrompt', '')[:100]}")

    elif msg_type in ("custom-title", "permission-mode", "agent-name"):
        print(f"[L{i}] [{msg_type}] {data.get(msg_type.replace('-', ' '), '')}")
