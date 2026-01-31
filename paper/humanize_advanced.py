#!/usr/bin/env python3
"""
Advanced Local Humanizer (Pegasus + Burstiness Optimization)
Target: 0% AI Detection via Entropy Maximization and Rhythm Variation.

This script:
1. Uses 'tuner007/pegasus_paraphrase' for structural rewriting (Abstractive).
2. Uses 'SentenceTransformers' to ensure meaning is preserved (Semantic Consistency).
3. Applies a 'Burstiness' algorithm: Enforces varied sentence lengths to mimic human writing rhythm.
4. Runs locally on your Mac (supports MPS/Metal acceleration).

Usage:
    python3 paper/humanize_advanced.py --input paper_content --iterations 1
"""

import os
import sys
import time
import torch
import nltk
import logging
import random
import argparse
import numpy as np
from importlib import import_module
from dataclasses import dataclass
from typing import List, Dict, Any

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Humanizer")

# Ensure NLTK data (for sentence splitting)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# ============================================================================
# Core Logic
# ============================================================================

class BurstinessEngine:
    """
    Selects the best paraphrase candidate based on:
    1. Semantic Similarity (must mean the same thing).
    2. Structural Distance (must look different).
    3. Rhythm Contrast (Short sentences follow long ones).
    """
    def __init__(self, device):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
    def select_best_candidate(self, original: str, candidates: List[str], prev_length: int) -> str:
        if not candidates:
            return original

        # 1. Semantic Similarity (We want high similarity to meaning)
        # Encode original and candidates
        embeddings = self.model.encode([original] + candidates)
        original_emb = embeddings[0]
        candidate_embs = embeddings[1:]
        
        # Cosine similarity
        from scipy.spatial.distance import cosine
        sim_scores = [1 - cosine(original_emb, cand) for cand in candidate_embs]

        # 2. Length Analysis (We want contrast with previous sentence)
        # If prev was long (>20 words), prefer short. If prev was short (<10), prefer long.
        target_len_type = "short" if prev_length > 20 else "long" if prev_length < 10 else "any"
        
        scored_candidates = []
        for i, cand in enumerate(candidates):
            sim = sim_scores[i]
            
            # Filter bad semantics
            if sim < 0.85: # If meaning changed too much, discard
                continue
                
            words = cand.split()
            curr_len = len(words)
            
            # Score Calculation
            score = sim * 1.0  # Base score is semantic match
            
            # Boost for length contrast (The "Burstiness" factor)
            if target_len_type == "short" and curr_len < 15:
                score += 0.15
            elif target_len_type == "long" and curr_len > 15:
                score += 0.15
            
            # Boost for vocabulary variation (Levenshtein ratio inverse roughly)
            # Simple check: Jaccard distance of words
            orig_set = set(original.lower().split())
            cand_set = set(cand.lower().split())
            if not orig_set.union(cand_set):
                 jaccard = 1.0
            else:
                 jaccard = len(orig_set.intersection(cand_set)) / len(orig_set.union(cand_set))
            
            # We want LOW Jaccard (different words) but HIGH Semantic Sim (same meaning)
            score += (1 - jaccard) * 0.2
            
            scored_candidates.append((score, cand))
        
        if not scored_candidates:
            return candidates[0] # Fallback
            
        # Sort by score desc
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return scored_candidates[0][1]

class PegasusHumanizer:
    def __init__(self):
        from transformers import PegasusForConditionalGeneration, PegasusTokenizer
        
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        # Fallback to cuda if linux/windows users copy this
        if torch.cuda.is_available(): self.device = "cuda"
            
        logger.info(f"Loading Pegasus on {self.device}...")
        
        model_name = "tuner007/pegasus_paraphrase"
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
        self.burst_engine = BurstinessEngine("cpu") # SentenceTransformer is light, CPU is fine
        logger.info("Models loaded successfully.")

    def humanize_sentence(self, sentence: str, prev_length: int) -> str:
        if len(sentence.split()) < 4: # Skip tiny fragments
            return sentence

        text = sentence
        batch = self.tokenizer([text], padding='longest', max_length=60, truncation=True, return_tensors="pt").to(self.device)
        
        # Generate varied candidates (High Temperature = High Entropy)
        translated = self.model.generate(
            **batch,
            max_length=60,
            num_beams=10, 
            num_return_sequences=5, 
            temperature=1.5,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        
        candidates = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        
        # Select best based on Burstiness
        best = self.burst_engine.select_best_candidate(sentence, candidates, prev_length)
        return best

    def humanize_paragraph(self, text: str) -> str:
        """
        Splits paragraph, humanizes sentences with rhythm awareness, rejoins.
        """
        sentences = nltk.sent_tokenize(text)
        new_sentences = []
        prev_len = 0
        
        for sent in sentences:
            new_sent = self.humanize_sentence(sent, prev_len)
            new_sentences.append(new_sent)
            prev_len = len(new_sent.split())
            
        return " ".join(new_sentences)

# ============================================================================
# Orchestration
# ============================================================================

def load_content(module_name: str) -> Dict[str, Any]:
    sys.path.insert(0, os.getcwd())
    module = import_module(module_name)
    return module.PAPER_CONTENT

def save_output(content: Dict[str, Any], filename: str):
    nl = chr(10)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('"""' + nl)
        f.write('Humanized Paper Content (Local Pegasus + Burstiness)' + nl)
        f.write(f'Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}' + nl + '"""' + nl + nl)
        
        f.write('PAPER_CONTENT = {' + nl)
        
        # Helper to dump fields
        def write_field(key, val, raw=False):
            if raw:
                # Use triple quotes for multiline raw strings
                val_esc = val.replace('"""', '\"\"\"')
                f.write(f'    "{key}": r"""{val_esc}""",{nl}{nl}')
            else:
                f.write(f'    "{key}": {repr(val)},{nl}{nl}')

        write_field("title", content["title"])
        write_field("author", content["author"], raw=True)
        write_field("abstract", content["abstract"], raw=True)
        write_field("keywords", content["keywords"])
        
        f.write('    "sections": [' + nl)
        for sec in content['sections']:
            f.write('        {' + nl)
            f.write(f'            "type": {repr(sec["type"])},{nl}')
            f.write(f'            "heading": {repr(sec["heading"])},{nl}')
            
            c_esc = sec["content"].replace('"""', '\"\"\"')
            f.write(f'            "content": r"""{c_esc}"""' + nl)
            f.write('        },' + nl)
        f.write('    ],' + nl + nl)
        
        write_field("bibliography", content["bibliography"], raw=True)
        f.write('}' + nl)
    
    logger.info(f"Saved to {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="paper_content_humanized_v2", help="Input python module")
    parser.add_argument("--output", default="paper_content_final.py", help="Output filename")
    args = parser.parse_args()
    
    logger.info(f"Initializing Local Humanizer (Input: {args.input})...")
    
    try:
        humanizer = PegasusHumanizer()
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.error("Did you run: pip install transformers sentencepiece sentence-transformers nltk scipy?")
        return

    content = load_content(args.input)
    
    # Process Abstract
    logger.info("Processing Abstract...")
    content['abstract'] = humanizer.humanize_paragraph(content['abstract'])
    
    # Process Sections
    total = len(content['sections'])
    for i, sec in enumerate(content['sections']):
        logger.info(f"Processing Section {i+1}/{total}: {sec['heading']}")
        if sec['content'].strip():
            sec['content'] = humanizer.humanize_paragraph(sec['content'])
            
    save_output(content, args.output)
    logger.info("Done. Now run your PDF generator on the new file.")

if __name__ == "__main__":
    main()
