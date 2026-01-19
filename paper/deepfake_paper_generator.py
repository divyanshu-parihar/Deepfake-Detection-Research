#!/usr/bin/env python3
"""
IEEE Paper Generator Pipeline.

This generator uses a Python module as the content source instead of JSON.
Python strings handle escaping naturally, making LaTeX content much easier to maintain.

Pipeline:
    paper_content.py (content) → deepfake_paper_generator.py → deepfake_paper.tex

Usage:
    python deepfake_paper_generator.py --module paper_content --output deepfake_paper.tex
"""
import os
import sys
import logging
import argparse
from importlib import import_module
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Default Constants
DEFAULT_OUTPUT_FILE = "deepfake_paper_humanized_v2.tex"
DEFAULT_CONTENT_MODULE = "paper_content_humanized_v2"

# IEEE Conference LaTeX Preamble
PREAMBLE = r"""\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{algorithm}
\usepackage{booktabs}
\usepackage{multirow}
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}
\begin{document}
"""


def generate_latex(content: dict[str, Any]) -> str:
    """
    Generate LaTeX document from content dictionary.
    
    Expected keys:
        title: str
        author: str (LaTeX formatted)
        abstract: str
        keywords: str
        sections: list[dict] with keys: type, heading, content
        bibliography: str
    """
    lines = [PREAMBLE]
    
    # Title
    if title := content.get("title"):
        lines.append(f"\\title{{{title}}}")
        lines.append("")
    
    # Author
    if author := content.get("author"):
        lines.append(f"\\author{{{author}}}")
        lines.append("")
    
    lines.append("\\maketitle")
    lines.append("")
    
    # Abstract
    if abstract := content.get("abstract"):
        lines.append("\\begin{abstract}")
        lines.append(abstract)
        lines.append("\\end{abstract}")
        lines.append("")
    
    # Keywords
    if keywords := content.get("keywords"):
        lines.append("\\begin{IEEEkeywords}")
        lines.append(keywords)
        lines.append("\\end{IEEEkeywords}")
        lines.append("")
    
    # Sections
    for sec in content.get("sections", []):
        sec_type = sec.get("type", "section")
        heading = sec.get("heading", "")
        body = sec.get("content", "")
        
        # Section command
        cmd = {
            "section": "\\section",
            "subsection": "\\subsection",
            "subsubsection": "\\subsubsection",
            "section*": "\\section*",
        }.get(sec_type, "\\section")
        
        lines.append(f"{cmd}{{{heading}}}")
        lines.append("")
        
        if body:
            lines.append(body)
            lines.append("")
    
    # Bibliography
    if bib := content.get("bibliography"):
        lines.append("\\begin{thebibliography}{00}")
        lines.append(bib)
        lines.append("\\end{thebibliography}")
    
    lines.append("")
    lines.append("\\end{document}")
    
    return "\n".join(lines)


def validate_latex(tex: str) -> list[str]:
    """Check for common LaTeX errors."""
    issues = []
    
    # Balanced braces
    if tex.count("{") != tex.count("}"):
        issues.append(f"Unbalanced braces: {tex.count('{')} open, {tex.count('}')} close")
    
    # Environment pairs
    import re
    for env in ["document", "abstract", "enumerate", "itemize", "table", 
                "tabular", "figure", "equation", "thebibliography"]:
        begins = len(re.findall(rf"\\begin\{{{env}}}", tex))
        ends = len(re.findall(rf"\\end\{{{env}}}", tex))
        if begins != ends:
            issues.append(f"Unbalanced {env}: {begins} begins, {ends} ends")
    
    return issues


def main() -> None:
    """Generate LaTeX from specified content module."""
    parser = argparse.ArgumentParser(description="Generate IEEE Paper LaTeX from Python content.")
    parser.add_argument("--module", default=DEFAULT_CONTENT_MODULE, help="Python module name containing PAPER_CONTENT (default: paper_content)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Output LaTeX filename (default: deepfake_paper.tex)")
    args = parser.parse_args()
    
    # Import content module
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        content_module = import_module(args.module)
        content = content_module.PAPER_CONTENT
    except ImportError as e:
        logger.error(f"Cannot import module '{args.module}': {e}")
        return
    except AttributeError:
        logger.error(f"Module '{args.module}' must define PAPER_CONTENT dict")
        return
    
    logger.info(f"Generating LaTeX from {args.module}...")
    latex = generate_latex(content)
    
    # Validate
    issues = validate_latex(latex)
    for issue in issues:
        logger.warning(issue)
    
    # Write output
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(latex)
        logger.info(f"Generated {args.output} ({len(latex)} bytes, {len(latex.splitlines())} lines)")
    except IOError as e:
        logger.error(f"Failed to write output file: {e}")

    if not issues:
        logger.info("Validation passed - ready to compile")


if __name__ == "__main__":
    main()
