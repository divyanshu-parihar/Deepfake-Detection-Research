#!/usr/bin/env python3
"""
Iterative Humanization System v3
Multi-API fallback with versioning and checkpoint/resume capability.

Features:
- Multi-API fallback chain (Decopy → AI-Text-Humanizer → Rephrasy)
- Versioned output: paper_content_v{N}.py
- JSON checkpoint for resume on failure
- Rate limiting with jitter
- Structured logging

Usage:
    python3 iterative_humanizer.py --iterations 3
    python3 iterative_humanizer.py --resume  # Resume from checkpoint
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import requests
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from importlib import import_module

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class HumanizerConfig:
    """System configuration"""
    max_iterations: int = 5
    delay_between_sections: int = 12
    delay_between_iterations: int = 30
    api_timeout: int = 60
    retry_attempts: int = 3
    checkpoint_file: str = "humanizer_checkpoint.json"
    input_module: str = "paper_content"
    output_prefix: str = "paper_content_v"


@dataclass
class Checkpoint:
    """Checkpoint state for resume capability"""
    current_version: int = 0
    current_iteration: int = 0
    sections_completed: List[str] = field(default_factory=list)
    sections_pending: List[str] = field(default_factory=list)
    abstract_done: bool = False
    last_updated: str = ""
    
    def save(self, path: str):
        self.last_updated = datetime.now().isoformat()
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.debug(f"Checkpoint saved: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'Checkpoint':
        if not os.path.exists(path):
            return cls()
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# ============================================================================
# API Clients
# ============================================================================

class BaseHumanizerAPI:
    """Base class for humanizer APIs"""
    name: str = "base"
    
    def humanize(self, text: str) -> Optional[str]:
        raise NotImplementedError


class DecopyAPI(BaseHumanizerAPI):
    """Decopy.ai humanizer (free tier has quotas)"""
    name = "Decopy.ai"
    BASE_URL = "https://api.decopy.ai/api/decopy/ai-humanizer"
    
    HEADERS = {
        'accept': '*/*',
        'content-type': 'application/x-www-form-urlencoded',
        'origin': 'https://decopy.ai',
        'referer': 'https://decopy.ai/',
        'product-serial': '2cd36e4c59ffd972baf9e8ff427db034',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    def __init__(self, timeout: int = 60):
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(self.HEADERS)
    
    def humanize(self, text: str) -> Optional[str]:
        if not text or len(text.strip()) < 50:
            return text
        
        # Create job
        try:
            resp = self._session.post(
                f"{self.BASE_URL}/create-job",
                data={
                    'entertext': text,
                    'length': 'expand',
                    'tone': 'academic',
                    'purpose': 'academic',
                    'language': 'English'
                },
                timeout=self.timeout
            )
            
            if resp.status_code != 200:
                logger.warning(f"[Decopy] HTTP {resp.status_code}")
                return None
            
            result = resp.json()
            if result.get('code') != 100000:
                msg = result.get('message', {})
                if 'Insufficient' in str(msg) or 'Quota' in str(msg):
                    logger.warning(f"[Decopy] Quota exhausted: {msg}")
                    return None
                logger.warning(f"[Decopy] API error: {msg}")
                return None
            
            job_id = result['result']['job_id']
            
        except Exception as e:
            logger.warning(f"[Decopy] Create job failed: {e}")
            return None
        
        # Poll for result
        time.sleep(10)
        for attempt in range(60):
            try:
                resp = self._session.get(
                    f"{self.BASE_URL}/get-job/{job_id}",
                    timeout=self.timeout
                )
                text_resp = resp.text.strip()
                
                if 'event: fail' in text_resp:
                    if 'data is not exists' in text_resp:
                        time.sleep(3)
                        continue
                    return None
                
                if 'event: message' in text_resp:
                    chunks = []
                    for line in text_resp.split('\n'):
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])
                                if data.get('state') == 100000 and 'data' in data:
                                    chunks.append(data['data'])
                            except json.JSONDecodeError:
                                pass
                    if chunks:
                        return ''.join(chunks)
                
                time.sleep(3)
                
            except Exception as e:
                logger.debug(f"[Decopy] Poll error: {e}")
                time.sleep(3)
        
        logger.warning("[Decopy] Job timed out")
        return None


class AITextHumanizerAPI(BaseHumanizerAPI):
    """AI-Text-Humanizer.com API"""
    name = "AI-Text-Humanizer"
    BASE_URL = "https://api.ai-text-humanizer.com"
    
    def __init__(self, timeout: int = 60):
        self.timeout = timeout
    
    def humanize(self, text: str) -> Optional[str]:
        if not text or len(text.strip()) < 50:
            return text
        
        try:
            resp = requests.post(
                f"{self.BASE_URL}/humanize",
                json={
                    "text": text,
                    "mode": "academic"
                },
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'Mozilla/5.0'
                },
                timeout=self.timeout
            )
            
            if resp.status_code == 200:
                data = resp.json()
                if 'humanized_text' in data:
                    return data['humanized_text']
                if 'result' in data:
                    return data['result']
            
            logger.debug(f"[AI-Text-Humanizer] HTTP {resp.status_code}")
            return None
            
        except Exception as e:
            logger.debug(f"[AI-Text-Humanizer] Error: {e}")
            return None


class RephrasifyAPI(BaseHumanizerAPI):
    """Rephrasy.ai humanizer API"""
    name = "Rephrasy"
    BASE_URL = "https://api.rephrasy.ai"
    
    def __init__(self, timeout: int = 60):
        self.timeout = timeout
    
    def humanize(self, text: str) -> Optional[str]:
        if not text or len(text.strip()) < 50:
            return text
        
        try:
            resp = requests.post(
                f"{self.BASE_URL}/v1/humanize",
                json={
                    "content": text,
                    "style": "academic"
                },
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'Mozilla/5.0'
                },
                timeout=self.timeout
            )
            
            if resp.status_code == 200:
                data = resp.json()
                if 'humanized' in data:
                    return data['humanized']
                if 'output' in data:
                    return data['output']
            
            logger.debug(f"[Rephrasy] HTTP {resp.status_code}")
            return None
            
        except Exception as e:
            logger.debug(f"[Rephrasy] Error: {e}")
            return None


# ============================================================================
# Main Humanizer
# ============================================================================

class VersionedHumanizer:
    """
    Multi-iteration humanization with versioning and checkpointing.
    """
    
    def __init__(self, config: HumanizerConfig):
        self.config = config
        self.checkpoint = Checkpoint.load(config.checkpoint_file)
        
        # API fallback chain
        self.apis: List[BaseHumanizerAPI] = [
            DecopyAPI(timeout=config.api_timeout),
            AITextHumanizerAPI(timeout=config.api_timeout),
            RephrasifyAPI(timeout=config.api_timeout),
        ]
        
        # Track stats
        self.stats = {
            'sections_humanized': 0,
            'api_failures': 0,
            'fallbacks_used': 0
        }
    
    def load_content(self, module_name: str) -> Dict[str, Any]:
        """Load paper content from Python module"""
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        module = import_module(module_name)
        return module.PAPER_CONTENT
    
    def humanize_text(self, text: str, heading: str) -> str:
        """Try each API in chain until success"""
        if not text or len(text.strip()) < 50:
            return text
        
        for idx, api in enumerate(self.apis):
            if idx > 0:
                self.stats['fallbacks_used'] += 1
                logger.info(f"  → Fallback to {api.name}")
            
            # Add jitter to avoid rate limits
            time.sleep(random.uniform(1, 3))
            
            result = api.humanize(text)
            if result:
                logger.info(f"  ✓ {api.name} succeeded ({len(result)} chars)")
                return result
            
            logger.warning(f"  ✗ {api.name} failed")
        
        # All APIs failed - return original
        self.stats['api_failures'] += 1
        logger.error(f"  ⚠ All APIs failed for: {heading[:30]}...")
        return text
    
    def run_iteration(self, content: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """Run one full iteration over all sections"""
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration}")
        logger.info(f"{'='*60}")
        
        output = {
            'title': content['title'],
            'author': content['author'],
            'keywords': content['keywords'],
            'bibliography': content['bibliography'],
            'abstract': content.get('abstract', ''),
            'sections': []
        }
        
        # Process abstract if not done in this iteration
        if not self.checkpoint.abstract_done or self.checkpoint.current_iteration < iteration:
            logger.info("\n[Abstract]")
            output['abstract'] = self.humanize_text(content['abstract'], "Abstract")
            self.checkpoint.abstract_done = True
            self.checkpoint.save(self.config.checkpoint_file)
            time.sleep(self.config.delay_between_sections)
        else:
            output['abstract'] = content['abstract']
            logger.info("[Abstract] Skipped (already done)")
        
        # Process sections
        sections = content.get('sections', [])
        total = len(sections)
        
        for idx, section in enumerate(sections):
            heading = section.get('heading', f'Section {idx}')
            sec_content = section.get('content', '')
            
            # Skip if already done in this iteration
            if heading in self.checkpoint.sections_completed and \
               self.checkpoint.current_iteration == iteration:
                output['sections'].append(section)
                logger.info(f"[{idx+1}/{total}] {heading[:40]}... Skipped")
                continue
            
            logger.info(f"\n[{idx+1}/{total}] {heading}")
            
            processed = section.copy()
            if sec_content and len(sec_content.strip()) > 50:
                processed['content'] = self.humanize_text(sec_content, heading)
                self.stats['sections_humanized'] += 1
            
            output['sections'].append(processed)
            
            # Update checkpoint
            if heading not in self.checkpoint.sections_completed:
                self.checkpoint.sections_completed.append(heading)
            self.checkpoint.save(self.config.checkpoint_file)
            
            # Rate limiting
            if idx < total - 1:
                delay = self.config.delay_between_sections + random.uniform(0, 5)
                logger.debug(f"  Waiting {delay:.1f}s...")
                time.sleep(delay)
        
        return output
    
    def save_version(self, content: Dict[str, Any], version: int):
        """Export content to versioned Python file"""
        output_path = f"{self.config.output_prefix}{version}.py"
        
        def escape_raw(text: str) -> str:
            return text.replace('"""', '\\"\\"\\"')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f'"""\nPaper Content - Humanized Version {version}\n')
            f.write(f'Generated: {datetime.now().isoformat()}\n"""\n\n')
            
            f.write('PAPER_CONTENT = {\n')
            f.write(f'    "title": {repr(content["title"])},\n\n')
            f.write(f'    "author": r"""{escape_raw(content["author"])}""",\n\n')
            f.write(f'    "abstract": r"""{escape_raw(content["abstract"])}""",\n\n')
            f.write(f'    "keywords": {repr(content["keywords"])},\n\n')
            
            f.write('    "sections": [\n')
            for section in content['sections']:
                f.write('        {\n')
                f.write(f'            "type": {repr(section["type"])},\n')
                f.write(f'            "heading": {repr(section["heading"])},\n')
                escaped = escape_raw(section.get('content', ''))
                f.write(f'            "content": r"""{escaped}"""\n')
                f.write('        },\n')
            f.write('    ],\n\n')
            
            f.write(f'    "bibliography": r"""{escape_raw(content["bibliography"])}"""\n')
            f.write('}\n')
        
        logger.info(f"\n✓ Saved: {output_path}")
        return output_path
    
    def run(self, resume: bool = False):
        """Run all iterations"""
        start_time = time.time()
        
        logger.info("\n" + "#"*60)
        logger.info("# ITERATIVE HUMANIZATION SYSTEM v3")
        logger.info("#"*60)
        logger.info(f"# Target iterations: {self.config.max_iterations}")
        logger.info(f"# Input: {self.config.input_module}")
        logger.info("#"*60 + "\n")
        
        # Load initial content
        start_version = self.checkpoint.current_version
        if resume and start_version > 0:
            # Resume from last version
            input_module = f"{self.config.output_prefix}{start_version}".replace('.py', '')
            logger.info(f"Resuming from version {start_version}")
        else:
            input_module = self.config.input_module
            start_version = 0
        
        try:
            content = self.load_content(input_module)
        except ImportError as e:
            logger.error(f"Failed to load {input_module}: {e}")
            return
        
        # Run iterations
        start_iter = self.checkpoint.current_iteration if resume else 1
        
        for iteration in range(start_iter, self.config.max_iterations + 1):
            self.checkpoint.current_iteration = iteration
            self.checkpoint.sections_completed = []
            self.checkpoint.abstract_done = False
            self.checkpoint.save(self.config.checkpoint_file)
            
            content = self.run_iteration(content, iteration)
            
            # Save this version
            version = start_version + iteration
            self.checkpoint.current_version = version
            self.save_version(content, version)
            self.checkpoint.save(self.config.checkpoint_file)
            
            # Delay between iterations
            if iteration < self.config.max_iterations:
                delay = self.config.delay_between_iterations
                logger.info(f"\nWaiting {delay}s before next iteration...")
                time.sleep(delay)
        
        # Summary
        elapsed = time.time() - start_time
        logger.info("\n" + "#"*60)
        logger.info("# COMPLETE")
        logger.info("#"*60)
        logger.info(f"# Sections humanized: {self.stats['sections_humanized']}")
        logger.info(f"# API fallbacks used: {self.stats['fallbacks_used']}")
        logger.info(f"# Total failures: {self.stats['api_failures']}")
        logger.info(f"# Total time: {elapsed/60:.1f} minutes")
        logger.info(f"# Final version: {self.config.output_prefix}{self.checkpoint.current_version}.py")
        logger.info("#"*60)
        logger.info("\nNext steps:")
        logger.info("1. Test output with AI detection tool (GPTZero, Turnitin)")
        logger.info("2. If not 0%, run again with: python3 iterative_humanizer.py --resume")
        logger.info("3. Generate LaTeX: python3 deepfake_paper_generator.py --module paper_content_vN")


def main():
    parser = argparse.ArgumentParser(
        description="Iterative Humanization System with versioning"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int, default=3,
        help="Number of humanization iterations (default: 3)"
    )
    parser.add_argument(
        "--input",
        type=str, default="paper_content",
        help="Input module name (default: paper_content)"
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--delay",
        type=int, default=12,
        help="Delay between sections in seconds (default: 12)"
    )
    
    args = parser.parse_args()
    
    config = HumanizerConfig(
        max_iterations=args.iterations,
        input_module=args.input,
        delay_between_sections=args.delay
    )
    
    humanizer = VersionedHumanizer(config)
    humanizer.run(resume=args.resume)


if __name__ == "__main__":
    main()
