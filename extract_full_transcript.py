"""
Extract readable conversation from Claude Code transcript JSONL file.
Outputs clean markdown with aligned tables and proper formatting.
"""
import json
import sys
from datetime import datetime

TRANSCRIPT = r"C:\Users\ChengYuxuan\OneDrive\ClaudeConfig\projects\D--ArtiIntComVis-MOSAIC\426427a3-fe5c-410f-a473-605e207a24e2.jsonl"
OUTPUT = r"C:\Users\ChengYuxuan\OneDrive\ClaudeConfig\note\frontres_transcript.md"

def quat_to_rpy_str(q):
    """Convert quaternion display to readable text."""
    return f"qw={q.get('qw',0):.3f}, qx={q.get('qx',0):.3f}, qy={q.get('qy',0):.3f}, qz={q.get('qz',0):.3f}"

def extract_text_from_content(content):
    """Extract user-facing text from a content array, skipping thinking/tool_use."""
    texts = []
    for item in content:
        if not isinstance(item, dict):
            continue
        t = item.get("type", "")
        if t == "text":
            texts.append(item.get("text", ""))
        elif t == "tool_use":
            # Show tool calls briefly
            name = item.get("name", "?")
            inp = item.get("input", {})
            desc = inp.get("description", "") or str(inp)[:120]
            texts.append(f"[Tool: {name}] {desc}")
        elif t == "tool_result":
            # Skip long tool results, show brief summary
            content_str = str(item.get("content", ""))
            is_error = item.get("is_error", False)
            if is_error:
                texts.append(f"[Tool Error] {content_str[:200]}")
            elif len(content_str) < 200:
                texts.append(f"[Tool Result] {content_str}")
            else:
                texts.append(f"[Tool Result] {content_str[:150]}...")
        # skip "thinking" type entirely
    return texts

def format_table(headers, rows, aligns=None):
    """Format a markdown table with aligned columns."""
    if not rows:
        return ""
    if aligns is None:
        aligns = ["l"] * len(headers)

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Build header
    sep = "|"
    for i, h in enumerate(headers):
        sep += " " + h.ljust(col_widths[i]) + " |"

    # Build alignment row
    align_row = "|"
    for i, a in enumerate(aligns):
        if a == "l":
            align_row += " " + ":" + "-" * (col_widths[i] - 1) + " |"
        elif a == "r":
            align_row += " " + "-" * (col_widths[i] - 1) + ": |"
        else:
            align_row += " " + ":" + "-" * (col_widths[i] - 2) + ": |"

    # Build data rows
    data_rows = []
    for row in rows:
        r = "|"
        for i, cell in enumerate(row):
            r += " " + str(cell).ljust(col_widths[i]) + " |"
        data_rows.append(r)

    return sep + "\n" + align_row + "\n" + "\n".join(data_rows)

def main():
    with open(TRANSCRIPT, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} lines from transcript...")

    with open(OUTPUT, "w", encoding="utf-8") as out:
        out.write("# FrontRES 对话完整记录\n\n")
        out.write(f"> 来源: {TRANSCRIPT}\n")
        out.write(f"> 行数: {len(lines)}\n\n")
        out.write("---\n\n")

        prev_date = ""
        msg_count = 0

        for line_num, line in enumerate(lines, 1):
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type", "")

            # Skip metadata entries
            if msg_type in ("last-prompt", "custom-title", "permission-mode",
                           "agent-name", "parentUuid"):
                continue

            # Get timestamp
            ts = data.get("timestamp", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    date_str = dt.strftime("%Y-%m-%d")
                    time_str = dt.strftime("%H:%M:%S")
                except Exception:
                    date_str = ""
                    time_str = ""
            else:
                date_str = ""
                time_str = ""

            # Date separator
            if date_str and date_str != prev_date:
                out.write(f"\n## {date_str}\n\n")
                prev_date = date_str

            if msg_type == "user":
                content = data.get("message", {}).get("content", [])
                texts = extract_text_from_content(content)
                if texts:
                    out.write(f"### YOU ({time_str})\n\n")
                    for t in texts:
                        # Skip pure tool results if they're just system output
                        if t.startswith("[Tool"):
                            continue
                        out.write(t + "\n\n")
                    out.write("---\n\n")
                    msg_count += 1

            elif msg_type == "assistant":
                content = data.get("message", {}).get("content", [])
                texts = extract_text_from_content(content)
                if texts:
                    out.write(f"### CLAUDE ({time_str})\n\n")
                    for t in texts:
                        out.write(t + "\n\n")
                    out.write("---\n\n")
                    msg_count += 1

        out.write(f"\n> Total messages extracted: {msg_count}\n")

    print(f"Done. {msg_count} messages written to:")
    print(f"  {OUTPUT}")

if __name__ == "__main__":
    main()
