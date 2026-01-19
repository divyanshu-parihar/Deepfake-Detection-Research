import re

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def extract_latex_components(tex_content):
    components = {}
    
    # Extract Preamble (everything up to \begin{document})
    preamble_match = re.search(r'(.*?)(\\begin{document})', tex_content, re.DOTALL)
    components['preamble'] = preamble_match.group(1) if preamble_match else ""
    
    # Extract Bibliography
    bib_match = re.search(r'(\\begin{{thebibliography}}.*?\\end{{thebibliography}})', tex_content, re.DOTALL)
    components['bibliography'] = bib_match.group(1) if bib_match else ""
    
    # Extract Figures
    components['figures'] = re.findall(r'(\\begin{figure}.*?\\end{figure})', tex_content, re.DOTALL)
    
    # Extract Tables
    components['tables'] = re.findall(r'(\\begin{table}.*?\\end{table})', tex_content, re.DOTALL)
    
    # Extract Equations
    components['equations'] = re.findall(r'(\\begin{equation}.*?\\end{equation})', tex_content, re.DOTALL)
    
    return components

def reconstruct_paper(clean_text, components):
    # This function is semi-heuristic because we need to map text back to structure
    
    new_tex = components['preamble'] + "\\begin{document}\n\n"
    
    # Title and Author (Hardcoded based on text content)
    new_tex += "\\title{Generalizable Deepfake Detection through Artifact-Invariant Representation Learning}\n\n"
    new_tex += "\\author{\\IEEEauthorblockN{Divyanshu Parihar}\n\\IEEEauthorblockA{\\textit{Independent Researcher} \\\
divyanshu1447@gmail.com}\n}\n\n\\maketitle\n\n"
    
    # Abstract
    abstract_text = re.search(r'abstract (.*?) abstract', clean_text, re.DOTALL | re.IGNORECASE)
    if not abstract_text: # Fallback if 'abstract' appears only once
         abstract_text = re.search(r'abstract (.*?) IEEEkeywords', clean_text, re.DOTALL | re.IGNORECASE)
    
    if abstract_text:
        new_tex += "\\begin{abstract}\n" + abstract_text.group(1).strip() + "\n\\end{abstract}\n\n"
    
    # Keywords
    keywords_text = re.search(r'IEEEkeywords (.*?) IEEEkeywords', clean_text, re.DOTALL | re.IGNORECASE)
    if not keywords_text:
        keywords_text = re.search(r'IEEEkeywords (.*?) Introduction', clean_text, re.DOTALL | re.IGNORECASE)

    if keywords_text:
        new_tex += "\\begin{IEEEkeywords}\n" + keywords_text.group(1).strip() + "\n\\end{IEEEkeywords}\n\n"
    
    # Main Content Processing
    # We'll split by known section headers found in the clean text
    # Note: I'm manually identifying headers from the clean text dump provided in the prompt
    
    # Remove the parts we already processed
    content_start = clean_text.find("Introduction")
    if content_start == -1: content_start = 0
    main_body = clean_text[content_start:]
    
    # Headers to look for (in order)
    headers = [
        ("Introduction", "section"),
        ("The Generalization Problem", "subsection"),
        ("Our Hypothesis", "subsection"),
        ("Contributions", "subsection"),
        ("Connected Studies", "section"), # Originally Related Work
        ("Early Detection Methods", "subsection"),
        ("Deep Learning Methods", "subsection"),
        ("Frequency-Domain Approaches", "subsection"),
        ("Attention and Transformer-Based Methods", "subsection"),
        ("Contrastive and Self-Supervised Learning", "subsection"),
        ("Limitations of Prior Work", "subsection"), # Or whatever it was renamed to
        ("Methodology", "section"),
        ("Overview", "subsection"),
        ("Frequency Stream Design", "subsection"),
        ("Grayscale Conversion", "subsubsection"),
        ("Block-wise DCT Computation", "subsubsection"),
        ("High-Pass Filtering", "subsubsection"),
        ("Frequency Feature Extraction", "subsubsection"),
        ("RGB Stream Design", "subsection"),
        ("Feature Fusion Architecture", "subsection"),
        ("Contrastive Learning Objective", "subsection"),
        ("Combined Training Objective", "subsection"),
        ("Training Procedure", "subsection"),
        ("Experimental Setup", "section"),
        ("Datasets", "subsection"),
        ("Data Preprocessing", "subsection"),
        ("Training Augmentations", "subsection"),
        ("Evaluation Protocol", "subsection"),
        ("Evaluation Metrics", "subsection"),
        ("Implementation Details", "subsection"),
        ("Baseline Methods", "subsection"),
        ("Results", "section"), # "The Results" in text?
        ("In-Domain Evaluation", "subsection"),
        ("Cross-Dataset Generalization", "subsection"),
        ("Evaluation on Additional Datasets", "subsection"),
        ("Ablation Studies", "subsection"),
        ("Effect of DCT Filter Cutoff", "subsection"),
        ("Contrastive Loss Analysis", "subsection"),
        ("Per-Manipulation Analysis", "subsection"),
        ("Robustness to Image Degradations", "subsection"),
        ("Computational Analysis", "subsection"),
        ("Additional Analysis", "section"),
        ("Visualization of Acquired Features", "subsection"),
        ("Embedding Space Analysis", "subsection"),
        ("Analysis of Failure Cases", "subsection"),
        ("Temporal Consistency", "subsection"),
        ("Discussion", "section"),
        ("Why Frequency Analysis Works", "subsection"), # "Discussion of Why..."
        ("Limitations and Future Work", "subsection"),
        ("Deployment Considerations", "subsection"),
        ("Conclusion", "section"),
        ("Acknowledgment", "section*")
    ]
    
    # We will iterate through headers, find them in text, and extract the content between them
    # We also need to inject figures/tables/equations
    
    current_pos = 0
    
    # Map section names in text (which might vary) to latex headers
    # I'll do a loose match
    
    normalized_body = main_body
    
    for i, (header, level) in enumerate(headers):
        # Find header position
        match = re.search(re.escape(header), normalized_body[current_pos:], re.IGNORECASE)
        if not match:
            # Try fuzzy match or alternative names seen in text
            if header == "Connected Studies": match = re.search("Connected Studies", normalized_body[current_pos:], re.IGNORECASE)
            elif header == "Results": match = re.search("The Results", normalized_body[current_pos:], re.IGNORECASE)
            elif header == "Why Frequency Analysis Works": match = re.search("Discussion of Why Frequency Analysis is Effective", normalized_body[current_pos:], re.IGNORECASE)
            elif header == "Visualization of Acquired Features": match = re.search("Visualization of Acquired Features", normalized_body[current_pos:], re.IGNORECASE)
            
        if match:
            # The content BEFORE this header belongs to the PREVIOUS section
            # (Or is the start of the Introduction if i=0)
            
            # If i > 0, we close the previous section content
            # But since we are building linearly, we just append content up to this match
            
            segment = normalized_body[current_pos : current_pos + match.start()]
            
            # PROCESS SEGMENT (Wrap lists, insert floats)
            segment = process_segment(segment, components)
            new_tex += segment + "\n\n"
            
            # Add the Header
            if level == "section*":
                new_tex += f"\\section*{{{header}}}\\n"
            else:
                new_tex += f"\\{level}{{{header}}}\\n"
            
            current_pos += match.end()
        else:
            print(f"Warning: Header '{header}' not found in text.")

    # Add remaining text (Conclusion/Ack body)
    segment = normalized_body[current_pos:]
    # Remove bibliography text artifacts if present at the end
    segment = re.split(r'00 \\bibitem', segment)[0] 
    
    new_tex += process_segment(segment, components) + "\n\n"
    
    # Add Bibliography
    new_tex += components['bibliography'] + "\n"
    
    new_tex += "\\end{document}"
    
    return new_tex

def process_segment(text, components):
    # 1. Lists
    # text has "enumerate ... enumerate" or "itemize ... itemize"
    # We replace the FIRST "enumerate" with \begin{enumerate} and the SECOND with \end{enumerate}
    # This is tricky if nested. Assuming flat lists based on chunks.
    
    # We can use regex to find pairs.
    # Pattern: enumerate (.*?) enumerate
    
    def repl_enum(m):
        content = m.group(1)
        # Add \item markers? The text usually has "1. " or just newlines.
        # Original text had "1. We introduce...", "2. We propose..."
        # If the clean text kept "1.", we can just wrap.
        # But commonly "enumerate" markers imply we need to format items.
        # Let's simple-wrap for now.
        return "\\begin{enumerate}\n" + content + "\n\\end{enumerate}"

    text = re.sub(r'enumerate\s*(.*?)\s*enumerate', repl_enum, text, flags=re.DOTALL)
    
    def repl_item(m):
        content = m.group(1)
        return "\\begin{itemize}\n" + content + "\n\\end{itemize}"

    text = re.sub(r'itemize\s*(.*?)\s*itemize', repl_item, text, flags=re.DOTALL)
    
    # 2. Figures and Tables
    # We need to heuristics to place them.
    # If text says "Fig. 1" or "Fig. [number]", we insert a figure.
    # If text says "Table [number]", we insert a table.
    
    # We'll pop from the list of components as we encounter references.
    # This assumes order is preserved (Introduction -> Conclusion).
    
    # Figures
    if "Fig." in text or "Figure" in text:
        # Check context. 
        # "Fig. 1" -> Figure 1
        # We can just insert the next available figure if we find a strong cue.
        # Or blindly insert specific figures if we can identify them.
        
        # Simple heuristic: If "Fig." is mentioned and we have figures left, insert one.
        # But we need to be careful not to dump all of them.
        pass # Too risky to automate without better cues. I will rely on standard float placement later if needed.
        # Better: Search for specific captions keywords in text?
        # "architecture" -> Fig 1
        # "DCT" -> Fig 2
        # "contrastive" -> Fig 3
        
        if "architecture" in text.lower() and len(components['figures']) >= 1:
            # Check if not already inserted? Simple script, so maybe we modify components list.
            # Let's find the fig with "architecture" in caption
            for i, fig in enumerate(components['figures']):
                if "architecture" in fig.lower():
                    text += "\n" + fig + "\n"
                    components['figures'].pop(i)
                    break
        
        if "DCT" in text and "filter" in text.lower():
             for i, fig in enumerate(components['figures']):
                if "DCT" in fig:
                    text += "\n" + fig + "\n"
                    components['figures'].pop(i)
                    break
                    
        if "contrastive" in text.lower() and "visualization" in text.lower():
             for i, fig in enumerate(components['figures']):
                if "contrastive" in fig.lower():
                    text += "\n" + fig + "\n"
                    components['figures'].pop(i)
                    break

    # Tables
    # Keywords: "In-Domain Evaluation", "Cross-Dataset", "Ablation", "Cutoff", "Contrastive Loss", "Per-Manipulation", "Robustness", "Computational"
    table_keywords = {
        "In-Domain Evaluation": "In-Domain",
        "Cross-Dataset Generalization": "Cross-Dataset",
        "Ablation Study": "Ablation",
        "Filter Cutoff": "Cutoff",
        "Contrastive Loss Weight": "Contrastive Loss",
        "Per-Manipulation": "Per-Manipulation",
        "Robustness": "Robustness",
        "Computational": "Computational"
    }
    
    for key, val in table_keywords.items():
        if key in text or val in text:
             for i, tbl in enumerate(components['tables']):
                if val in tbl: # Check if caption contains keyword
                    text += "\n" + tbl + "\n"
                    components['tables'].pop(i)
                    break
    
    # Equations
    # Text usually has "equation" marker or empty gaps.
    # "equation" marker -> Insert equation.
    
    if "equation" in text:
        # Replace first "equation" word with first equation
        if len(components['equations']) > 0:
             text = text.replace("equation", "\n" + components['equations'][0] + "\n", 1)
             components['equations'].pop(0)
    
    # Also handle the "where when , and otherwise" mangled equation
    if "where when , and otherwise" in text and len(components['equations']) > 0:
         # This usually corresponds to the DCT equation or the filter equation.
         # Actually "where ... when ... otherwise" sounds like the filter equation (Eq 3).
         # Eq 2 is DCT formula.
         # Let's checking Eq 2 in original text: "where C(u)=...".
         # Eq 3 is Filter: "F'(u,v) = ... cases ...".
         pass

    # Fallback: Just dump remaining equations if "equation" word is found again
    while "equation" in text and len(components['equations']) > 0:
         text = text.replace("equation", "\n" + components['equations'].pop(0) + "\n", 1)

    return text

def main():
    tex_content = read_file('paper/deepfake_paper.tex')
    
    # Load all chunks and join them
    clean_text = read_file('paper/clean_text_chunks.txt')
    # Remove the "--- Chunk X ---" markers
    clean_text = re.sub(r'--- Chunk \d+ ---', '', clean_text)
    
    components = extract_latex_components(tex_content)
    
    new_tex = reconstruct_paper(clean_text, components)
    
    with open('paper/deepfake_paper.tex', 'w') as f:
        f.write(new_tex)
    
    print("Reconstruction complete.")

if __name__ == "__main__":
    main()
