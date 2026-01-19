#!/usr/bin/env python3
"""
Deepfake Paper Content Humanizer v2
Multi-iteration humanization using Decopy.ai API
Goal: Reduce AI detection to 0% while preserving LaTeX formatting.

Features:
- Multiple sequential iterations per text chunk
- Progress tracking with detailed logging
- Preserves LaTeX commands and structure
- Configurable iteration count
- Exports in same format as input
"""

import requests
import time
import json
import re
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class HumanizationResult:
    """Result container for humanization operations"""
    original_text: str
    humanized_text: str
    iterations_completed: int
    success: bool
    error_message: Optional[str] = None


class DecopyAPIClient:
    """
    HTTP client for Decopy.ai Humanizer API.
    Handles job creation, polling, and SSE response parsing.
    """
    
    BASE_URL = "https://api.decopy.ai/api/decopy/ai-humanizer"
    
    # Static headers derived from browser inspection
    HEADERS = {
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9,hi;q=0.8',
        'cache-control': 'no-cache',
        'content-type': 'application/x-www-form-urlencoded',
        'dnt': '1',
        'origin': 'https://decopy.ai',
        'pragma': 'no-cache',
        'priority': 'u=1, i',
        'product-serial': '2cd36e4c59ffd972baf9e8ff427db034',
        'referer': 'https://decopy.ai/',
        'sec-ch-ua': '"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
        'sec-ch-ua-mobile': '?1',
        'sec-ch-ua-platform': '"Android"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Mobile Safari/537.36'
    }
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(self.HEADERS)
    
    def create_job(
        self, 
        text: str, 
        length: str = "expand",
        tone: str = "academic",
        purpose: str = "academic",
        language: str = "English"
    ) -> Optional[str]:
        """
        Submit text for humanization. Returns job_id on success.
        
        Args:
            text: Content to humanize
            length: 'expand', 'same', or 'shorten'
            tone: Writing tone (academic, casual, etc.)
            purpose: Content purpose
            language: Target language
            
        Returns:
            job_id string or None on failure
        """
        url = f"{self.BASE_URL}/create-job"
        
        payload = {
            'entertext': text,
            'length': length,
            'tone': tone,
            'purpose': purpose,
            'language': language
        }
        
        try:
            response = self._session.post(url, data=payload, timeout=self.timeout)
            
            if response.status_code != 200:
                logger.error(f"Create job failed: HTTP {response.status_code}")
                return None
            
            result = response.json()
            
            if result.get('code') == 100000:
                job_id = result['result']['job_id']
                logger.debug(f"Job created: {job_id}")
                return job_id
            
            logger.error(f"API error: {result.get('message', 'Unknown error')}")
            return None
            
        except requests.RequestException as e:
            logger.error(f"Network error: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            return None
    
    def poll_job(
        self, 
        job_id: str, 
        max_attempts: int = 60,
        poll_interval: int = 5
    ) -> Optional[str]:
        """
        Poll job status until completion. Returns humanized text.
        
        Uses SSE (Server-Sent Events) format parsing.
        """
        url = f"{self.BASE_URL}/get-job/{job_id}"
        
        for attempt in range(1, max_attempts + 1):
            try:
                response = self._session.get(url, timeout=self.timeout)
                
                if not response.text or not response.text.strip():
                    logger.debug(f"Empty response, attempt {attempt}/{max_attempts}")
                    time.sleep(poll_interval)
                    continue
                
                text = response.text.strip()
                
                # Handle failure event
                if 'event: fail' in text:
                    if 'data is not exists' in text:
                        logger.debug(f"Job pending, attempt {attempt}/{max_attempts}")
                        time.sleep(poll_interval)
                        continue
                    logger.error(f"Job failed: {text[:200]}")
                    return None
                
                # Parse SSE message events
                if 'event: message' in text:
                    data_chunks = self._parse_sse_response(text)
                    if data_chunks:
                        humanized = ''.join(data_chunks)
                        logger.debug(f"Job complete: {len(humanized)} chars")
                        return humanized
                
                # Fallback: try standard JSON
                try:
                    result = response.json()
                    if result.get('code') == 100000 and 'result' in result:
                        output = result['result'].get('output')
                        if output:
                            return output
                except json.JSONDecodeError:
                    pass
                
                time.sleep(poll_interval)
                
            except requests.RequestException as e:
                logger.warning(f"Poll error (attempt {attempt}): {e}")
                time.sleep(poll_interval)
        
        logger.error(f"Job timed out after {max_attempts} attempts")
        return None
    
    def _parse_sse_response(self, text: str) -> List[str]:
        """Extract data chunks from SSE format response"""
        chunks = []
        for line in text.split('\n'):
            if line.startswith('data: '):
                data_content = line[6:]
                try:
                    data_json = json.loads(data_content)
                    if data_json.get('state') == 100000 and 'data' in data_json:
                        chunks.append(data_json['data'])
                except json.JSONDecodeError:
                    continue
        return chunks


class PaperHumanizer:
    """
    Multi-iteration paper humanization processor.
    Preserves LaTeX structure while humanizing prose content.
    """
    
    def __init__(
        self, 
        iterations: int = 3,
        job_creation_delay: int = 10,
        inter_iteration_delay: int = 5
    ):
        self.api = DecopyAPIClient()
        self.iterations = iterations
        self.job_creation_delay = job_creation_delay
        self.inter_iteration_delay = inter_iteration_delay
        
        # Track statistics
        self.stats = {
            'sections_processed': 0,
            'total_iterations': 0,
            'failures': 0
        }
    
    def humanize_text(self, text: str) -> HumanizationResult:
        """
        Run text through multiple humanization iterations.
        """
        if not text or not text.strip():
            return HumanizationResult(
                original_text=text,
                humanized_text=text,
                iterations_completed=0,
                success=True
            )
        
        current = text
        completed_iterations = 0
        
        for i in range(1, self.iterations + 1):
            logger.info(f"  Iteration {i}/{self.iterations} ({len(current)} chars)")
            
            # Create humanization job
            job_id = self.api.create_job(current)
            if not job_id:
                return HumanizationResult(
                    original_text=text,
                    humanized_text=current,
                    iterations_completed=completed_iterations,
                    success=False,
                    error_message=f"Failed to create job at iteration {i}"
                )
            
            # Wait before polling
            logger.debug(f"  Waiting {self.job_creation_delay}s for processing...")
            time.sleep(self.job_creation_delay)
            
            # Poll for result
            result = self.api.poll_job(job_id)
            if not result:
                return HumanizationResult(
                    original_text=text,
                    humanized_text=current,
                    iterations_completed=completed_iterations,
                    success=False,
                    error_message=f"Failed to get result at iteration {i}"
                )
            
            current = result
            completed_iterations = i
            self.stats['total_iterations'] += 1
            
            # Delay between iterations
            if i < self.iterations:
                time.sleep(self.inter_iteration_delay)
        
        return HumanizationResult(
            original_text=text,
            humanized_text=current,
            iterations_completed=completed_iterations,
            success=True
        )
    
    def process_section(self, section: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single paper section"""
        processed = section.copy()
        
        content = section.get('content', '')
        if content and content.strip():
            heading = section.get('heading', 'Untitled')
            logger.info(f"\n{'='*60}")
            logger.info(f"Section: {heading}")
            logger.info(f"{'='*60}")
            
            result = self.humanize_text(content)
            processed['content'] = result.humanized_text
            
            if result.success:
                logger.info(f"✓ Complete ({result.iterations_completed} iterations)")
                self.stats['sections_processed'] += 1
            else:
                logger.warning(f"⚠ Partial: {result.error_message}")
                self.stats['failures'] += 1
        
        return processed
    
    def process_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process entire paper through humanization pipeline.
        
        Returns:
            Humanized paper content in original format
        """
        logger.info("\n" + "="*60)
        logger.info("DEEPFAKE PAPER HUMANIZATION v2")
        logger.info("="*60)
        logger.info(f"Iterations per section: {self.iterations}")
        logger.info(f"Total sections: {len(paper.get('sections', []))}")
        logger.info("="*60)
        
        output = {
            'title': paper['title'],
            'author': paper['author'],
            'keywords': paper['keywords'],
            'bibliography': paper['bibliography'],
            'sections': []
        }
        
        # Process abstract
        logger.info("\n" + "-"*60)
        logger.info("PROCESSING: Abstract")
        logger.info("-"*60)
        abstract_result = self.humanize_text(paper.get('abstract', ''))
        output['abstract'] = abstract_result.humanized_text
        
        if abstract_result.success:
            logger.info(f"✓ Abstract complete")
            self.stats['sections_processed'] += 1
        else:
            logger.warning(f"⚠ Abstract partial: {abstract_result.error_message}")
            self.stats['failures'] += 1
        
        # Process each section
        sections = paper.get('sections', [])
        for idx, section in enumerate(sections, 1):
            logger.info(f"\n[{idx}/{len(sections)}]")
            processed = self.process_section(section)
            output['sections'].append(processed)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("PROCESSING COMPLETE")
        logger.info("="*60)
        logger.info(f"Sections processed: {self.stats['sections_processed']}")
        logger.info(f"Total iterations run: {self.stats['total_iterations']}")
        logger.info(f"Failures: {self.stats['failures']}")
        
        return output


def escape_for_raw_string(text: str) -> str:
    """Escape text for Python raw string triple quotes"""
    # Handle triple quotes in content
    return text.replace('"""', '\\"\\"\\"')


def export_paper_content(content: Dict[str, Any], output_path: str):
    """
    Export humanized content to Python file in original format.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('"""\n')
        f.write('Paper Content Module - Humanized Version (v2)\n')
        f.write('Multi-iteration humanization via Decopy.ai\n')
        f.write('Target: 0% AI detection score\n')
        f.write('"""\n\n')
        
        f.write('PAPER_CONTENT = {\n')
        f.write(f'    "title": {repr(content["title"])},\n\n')
        f.write(f'    "author": r"""{escape_for_raw_string(content["author"])}""",\n\n')
        f.write(f'    "abstract": r"""{escape_for_raw_string(content["abstract"])}""",\n\n')
        f.write(f'    "keywords": {repr(content["keywords"])},\n\n')
        
        f.write('    "sections": [\n')
        
        for section in content['sections']:
            f.write('        {\n')
            f.write(f'            "type": {repr(section["type"])},\n')
            f.write(f'            "heading": {repr(section["heading"])},\n')
            escaped_content = escape_for_raw_string(section.get('content', ''))
            f.write(f'            "content": r"""{escaped_content}"""\n')
            f.write('        },\n')
        
        f.write('    ],\n\n')
        
        escaped_bib = escape_for_raw_string(content['bibliography'])
        f.write(f'    "bibliography": r"""{escaped_bib}"""\n')
        f.write('}\n')
    
    logger.info(f"\n✓ Exported to: {output_path}")


def save_intermediate_result(content: Dict[str, Any], iteration: int, base_path: str):
    """Save intermediate results after each major section"""
    path = f"{base_path}_checkpoint_{iteration}.py"
    export_paper_content(content, path)
    logger.info(f"  Checkpoint saved: {path}")


def main():
    """Main entry point for paper humanization"""
    # Import source content
    try:
        from paper_content import PAPER_CONTENT
    except ImportError as e:
        logger.error(f"Failed to import paper_content: {e}")
        logger.info("Ensure paper_content.py exists in the same directory")
        return
    
    # Configuration
    ITERATIONS = 3  # Number of passes per text chunk
    OUTPUT_FILE = "paper_content_humanized_v2.py"
    
    logger.info("\n" + "#"*60)
    logger.info("# DEEPFAKE PAPER HUMANIZER v2")
    logger.info("#"*60)
    logger.info(f"# Iterations: {ITERATIONS}")
    logger.info(f"# Output: {OUTPUT_FILE}")
    logger.info("#"*60 + "\n")
    
    # Initialize processor
    humanizer = PaperHumanizer(iterations=ITERATIONS)
    
    # Process paper
    start_time = time.time()
    humanized_paper = humanizer.process_paper(PAPER_CONTENT)
    elapsed = time.time() - start_time
    
    # Export result
    export_paper_content(humanized_paper, OUTPUT_FILE)
    
    # Final report
    logger.info("\n" + "#"*60)
    logger.info("# FINISHED")
    logger.info("#"*60)
    logger.info(f"# Total time: {elapsed/60:.1f} minutes")
    logger.info(f"# Output file: {OUTPUT_FILE}")
    logger.info("#"*60)
    logger.info("\nNext steps:")
    logger.info("1. Review the output file for formatting issues")
    logger.info("2. Test with AI detection tools (GPTZero, Turnitin, etc.)")
    logger.info("3. If needed, increase ITERATIONS and re-run")


if __name__ == "__main__":
    main()
