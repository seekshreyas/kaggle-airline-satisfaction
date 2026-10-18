#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal builder for Deckset-flavored markdown (https://docs.deckset.com).

- global commands go at the top, with no blank lines between them
- slides are separated by `---`
- paragraphs starting with `^` are presenter notes
- images: ![<options> alt(<description>)](path)
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

from dataclasses import dataclass, field


def image(path: str, alt: str, options: str = "inline fill") -> str:
    """An informative image with an accessible description."""
    return f"![{options} alt({alt})]({path})"


def notes(*paragraphs: str) -> str:
    """Presenter notes: one `^` paragraph per argument."""
    return "\n\n".join(f"^ {p.strip()}" for p in paragraphs if p and p.strip())


def bullets(*items: str) -> str:
    return "\n".join(f"- {i}" for i in items)


def columns(*cols: str) -> str:
    """Equal-width columns."""
    return "\n\n".join(f"[.column]\n\n{c.strip()}" for c in cols)


@dataclass
class Deck:
    theme: str | None = None
    footer: str | None = None
    slidenumbers: bool = True
    autoscale: bool = True
    slides: list[str] = field(default_factory=list)

    def slide(self, *blocks: str, commands: tuple[str, ...] = ()) -> "Deck":
        """Add a slide made of markdown blocks (per-slide commands like '[.hide-footer]' first)."""
        body = [*commands, *[b.strip() for b in blocks if b and b.strip()]]
        self.slides.append("\n\n".join(body))
        return self

    def header(self) -> str:
        cmds = []
        if self.theme:
            cmds.append(f"theme: {self.theme}")
        if self.footer:
            cmds.append(f"footer: {self.footer}")
        cmds.append(f"slidenumbers: {str(self.slidenumbers).lower()}")
        cmds.append(f"autoscale: {str(self.autoscale).lower()}")
        return "\n".join(cmds)  # no blank lines between global commands

    def render(self) -> str:
        return self.header() + "\n\n" + "\n\n---\n\n".join(self.slides) + "\n"
