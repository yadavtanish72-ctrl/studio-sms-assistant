"""content/*.md -> 26 chunks, pure Python with no network or keys.

One chunk per `##` section prefixed with "Doc Title > Heading", except the atomic
schedule; ids are `<file>#<heading-slug>`, the form eval/golden.yaml cites.
"""
import re
from dataclasses import dataclass

import yaml

from . import config

DAY_NAMES = {"Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday",
             "Fri": "Friday", "Sat": "Saturday", "Sun": "Sunday"}


def slugify(heading):
    """'Level 1 - Foundations' -> 'level-1-foundations'. eval/audit_golden.py keeps an
    identical copy so it can validate golden.yaml without importing the pipeline."""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9 -]", "", heading.lower()).replace(" ", "-"))


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit. `content` is embedded, BM25-indexed AND shown to the
    generator, so anything added for retrieval must also be true."""
    id: str
    content: str          # embedded, indexed for BM25, and shown to the generator
    source_file: str
    heading_path: str
    is_atomic: bool


def _split_frontmatter(text):
    """Return (frontmatter dict, body), raising if the block is missing."""
    m = re.match(r"---\n(.*?)\n---\n(.*)", text, re.S)
    if not m:
        raise ValueError("file has no YAML frontmatter")
    return yaml.safe_load(m.group(1)), m.group(2)


def _informal_time(t):
    """'6:00 AM' -> '6am'; '6:30 AM' -> '6:30am'. The bare-hour form is only emitted for
    on-the-hour times, so no time is ever invented."""
    hour, minute, meridiem = re.match(r"(\d{1,2}):(\d{2}) (AM|PM)", t).groups()
    stem = hour if minute == "00" else f"{hour}:{minute}"
    return stem + meridiem.lower()


def _schedule_sentences(body):
    """Rewrite the table as sentences and spell out day abbreviations (PLAN §7a).
    Without this BM25 cannot match any day or informal time against the schedule."""
    out = []
    for line in body.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[0] in DAY_NAMES:
            day, time, cls, who, level, mins = cells
            out.append(f"{DAY_NAMES[day]} {time} ({_informal_time(time)}): "
                       f"{cls} with {who}, Level {level}, {mins} minutes.")
        elif line.startswith("|"):
            continue  # header and separator rows
        else:
            out.append(line)  # surrounding prose is kept as-is
    return "\n".join(out)


def chunk_file(path):
    """One markdown file -> its chunks. Atomic files yield exactly one."""
    meta, body = _split_frontmatter(path.read_text())
    title = meta["title"]
    body = re.sub(r"^# .+$", "", body, count=1, flags=re.M)  # H1 duplicates the title

    # The atomic path: schedule.md sets `atomic: true` in its frontmatter. Cited as its H1
    # slug, since it has no `##` sections to name.
    if meta.get("atomic"):
        heading_path = title
        return [Chunk(id=f"{path.name}#{slugify(title)}",
                      content=f"{heading_path}\n\n{_schedule_sentences(body).strip()}",
                      source_file=path.name, heading_path=heading_path, is_atomic=True)]

    # re.split with a capturing group interleaves [before, heading, body, heading, body...],
    # so parts[1::2] are the headings and parts[2::2] their sections.
    parts = re.split(r"^## (.+)$", body, flags=re.M)
    # parts[0] is whatever preceded the first '##', and should be empty once the H1 is gone.
    # Refusing to continue is the point: prose here would be dropped from the index unnoticed.
    if parts[0].strip():
        raise ValueError(f"{path.name}: prose before the first '##' would be dropped: "
                         f"{parts[0].strip()[:60]!r}")
    chunks = []
    for heading, section in zip(parts[1::2], parts[2::2]):
        heading_path = f"{title} > {heading.strip()}"
        chunks.append(Chunk(id=f"{path.name}#{slugify(heading)}",
                            content=f"{heading_path}\n\n{section.strip()}",
                            source_file=path.name, heading_path=heading_path, is_atomic=False))
    return chunks


def all_chunks():
    """The whole corpus, in stable filename order. 26 chunks as of today's content/."""
    chunks = [c for p in sorted(config.CONTENT_DIR.glob("*.md")) for c in chunk_file(p)]
    # Ids are the primary key in Pinecone: a duplicate would mean one chunk silently
    # overwriting another at upsert, leaving the index short with no error anywhere.
    ids = [c.id for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate chunk ids: {sorted({i for i in ids if ids.count(i) > 1})}")
    return chunks
