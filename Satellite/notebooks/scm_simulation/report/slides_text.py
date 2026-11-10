"""The text of every slide, one file per slide in deck order, for the number checker and for reading the deck as text.

Each paragraph, heading and note is one line (a note starts with NOTES:), each table row is its cells joined by " | ",
and an image is [image: its alt text].

    python report/slides_text.py   rewrites report/deck/slides_text/NN_<slide id>.txt
"""
import json
from html.parser import HTMLParser
from pathlib import Path

DECK = Path(__file__).resolve().parent / "deck"
BLOCK = {"p", "h1", "h2", "h3", "aside"}


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.lines, self.buf, self.row, self.cell = [], None, None, None

    def handle_starttag(self, tag, attrs):
        if tag in BLOCK:
            self.buf = []
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = []
        elif tag == "img":
            self.lines.append(f"[image: {dict(attrs).get('alt', '')}]")

    def handle_endtag(self, tag):
        if tag in BLOCK and self.buf is not None:
            t = "".join(self.buf).strip()
            self.lines.append(("NOTES: " + t) if tag == "aside" else t)
            self.buf = None
        elif tag in ("td", "th") and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.lines.append((" | ".join(self.row) + " |").strip())
            self.row = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)
        elif self.buf is not None:
            self.buf.append(data)


def main():
    order = json.loads((DECK / "project" / "deck.json").read_text())["order"]
    out = DECK / "slides_text"
    for p in out.glob("*.txt"):
        p.unlink()
    for i, sid in enumerate(order, 1):
        t = Text()
        t.feed((DECK / "project" / "slides" / f"{sid}.html").read_text())
        (out / f"{i:02d}_{sid}.txt").write_text("\n".join(t.lines) + "\n")
    print(f"{len(order)} slides written to {out}")


if __name__ == "__main__":
    main()
