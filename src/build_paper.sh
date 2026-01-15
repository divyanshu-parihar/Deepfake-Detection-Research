#!/bin/bash
# Script to compile the research paper
# Requires pdflatex

if command -v pdflatex &> /dev/null; then
    echo "Compiling LaTeX..."
    pdflatex -interaction=nonstopmode paper/deepfake_paper.tex
    pdflatex -interaction=nonstopmode paper/deepfake_paper.tex # Run twice for references
    echo "Done! Output is deepfake_paper.pdf"
else
    echo "Error: pdflatex not found. Please install a LaTeX distribution (e.g., MacTeX)."
    echo "The .tex file has been updated successfully."
fi