import re

def clean_latex_aggressive(tex_content):
    # 1. Remove comments
    text = re.sub(r'%.*', '', tex_content)

    # 2. Extract content between \begin{document} and \end{document}
    doc_match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', text, re.DOTALL)
    if doc_match:
        text = doc_match.group(1)
    
    # 3. Remove bibliography section entirely
    text = re.sub(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}', '', text, flags=re.DOTALL)

    # 4. Remove all math environments
    text = re.sub(r'\$.*?\$', '', text)
    text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{(?:equation|align|displaymath|eqnarray|math)\*?\}.*?\\end\{.*?\}', '', text, flags=re.DOTALL)

    # 5. Remove Figures, Tables, and Algorithms
    text = re.sub(r'\\begin\{(?:figure|table|algorithm|tabular|center)\*?\}.*?\\end\{.*?\}', '', text, flags=re.DOTALL)

    # 6. Remove specific structural commands but KEEP their text content
    content_commands = ['section', 'subsection', 'subsubsection', 'title', 'author', 'abstract', 'textit', 'textbf', 'emph', 'IEEEauthorblockN', 'IEEEauthorblockA']
    for cmd in content_commands:
        pattern = r'\\' + cmd + r'\*?\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        while re.search(pattern, text):
            text = re.sub(pattern, r' \1 ', text)

    # 7. Remove commands that should be DELETED entirely
    delete_commands = ['cite', 'label', 'ref', 'eqref', 'bibliographystyle', 'bibliography', 'IEEEkeywords', 'maketitle', 'IEEEoverridecommandlockouts', 'usepackage', 'def', 'item']
    for cmd in delete_commands:
        pattern = r'\\' + cmd + r'\*?\{.*?\}'
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    
    # 8. Remove standalone backslash commands
    text = re.sub(r'\\[a-zA-Z]+\*?', ' ', text)

    # 9. Clean up LaTeX special characters
    text = text.replace('``', '"').replace("''", '"')
    text = text.replace('---', ' — ').replace('--', ' – ')
    text = text.replace('~', ' ')
    
    # 10. Final pass: remove any remaining braces and backslashes
    text = re.sub(r'[\\\{\}]', '', text)
    
    # 11. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def split_into_chunks(text, words_per_chunk=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunk = words[i:i + words_per_chunk]
        chunks.append(' '.join(chunk))
    return chunks

def main():
    try:
        with open('paper/deepfake_paper.tex', 'r') as f:
            content = f.read()
        
        cleaned_text = clean_latex_aggressive(content)
        chunks = split_into_chunks(cleaned_text, 300)
        
        output_filename = 'paper/clean_text_chunks.txt'
        with open(output_filename, 'w') as f:
            for i, chunk in enumerate(chunks):
                f.write(f"--- Chunk {i+1} ---\n")
                f.write(chunk)
                f.write("\n\n")
                
        print(f"Successfully processed {len(chunks)} clean chunks to {output_filename}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
