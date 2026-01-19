#!/usr/bin/env python3
"""
Academic Paper Humanizer using Decopy.ai API
Processes paper content through multiple humanization iterations
to achieve 0% AI plagiarism detection while maintaining format.
"""

import requests
import time
import json
import re
from typing import Dict, Any, Optional
from urllib.parse import urlencode

class PaperHumanizer:
    def __init__(self):
        self.base_url = "https://api.decopy.ai/api/decopy/ai-humanizer"
        self.headers = {
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
    
    def create_job(self, text: str, length: str = "expand", 
                   tone: str = "academic", purpose: str = "academic",
                   language: str = "English") -> Optional[str]:
        """
        Create a humanization job and return job_id
        """
        url = f"{self.base_url}/create-job"
        
        # Prepare form data
        data = {
            'entertext': text,
            'length': length,
            'tone': tone,
            'purpose': purpose,
            'language': language
        }
        
        try:
            response = requests.post(url, headers=self.headers, data=data, timeout=30)
            
            # Debug output
            print(f"  Create job status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"  Response: {response.text[:200]}")
                return None
            
            result = response.json()
            
            if result.get('code') == 100000:
                job_id = result['result']['job_id']
                print(f"✓ Job created: {job_id}")
                return job_id
            else:
                print(f"✗ Job creation failed: {result}")
                return None
                
        except requests.RequestException as e:
            print(f"✗ Network error creating job: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"✗ Invalid JSON response: {e}")
            print(f"  Response text: {response.text[:200]}")
            return None
        except Exception as e:
            print(f"✗ Unexpected error creating job: {e}")
            return None
    
    def get_job_result(self, job_id: str, max_retries: int = 60, 
                       retry_delay: int = 5) -> Optional[str]:
        """
        Poll for job completion and return humanized text
        The API returns Server-Sent Events (SSE) format
        """
        url = f"{self.base_url}/get-job/{job_id}"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=30)
                
                # Debug: Print raw response on first attempt
                if attempt == 0:
                    print(f"  Response status: {response.status_code}")
                
                # Check if response has content
                if not response.text or response.text.strip() == '':
                    print(f"  Empty response (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                
                # Parse SSE format response
                text = response.text.strip()
                
                # Check for failure event
                if 'event: fail' in text:
                    if 'data is not exists' in text:
                        print(f"  Job not ready yet (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"✗ Job failed: {text}")
                        return None
                
                # Check for message event with data
                if 'event: message' in text:
                    # Extract all data lines
                    data_lines = []
                    for line in text.split('\n'):
                        if line.startswith('data: '):
                            data_content = line[6:]  # Remove 'data: ' prefix
                            try:
                                data_json = json.loads(data_content)
                                if data_json.get('state') == 100000 and 'data' in data_json:
                                    data_lines.append(data_json['data'])
                            except json.JSONDecodeError:
                                continue
                    
                    # If we got data, combine it
                    if data_lines:
                        humanized_text = ''.join(data_lines)
                        print(f"✓ Job completed successfully ({len(humanized_text)} chars)")
                        return humanized_text
                    else:
                        print(f"  Partial data received (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                
                # Try standard JSON format as fallback
                try:
                    result = response.json()
                    if result.get('code') == 100000 and 'result' in result:
                        if 'output' in result['result']:
                            humanized_text = result['result']['output']
                            print(f"✓ Job completed successfully")
                            return humanized_text
                except json.JSONDecodeError:
                    pass
                
                print(f"  Waiting for completion (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                    
            except requests.RequestException as e:
                print(f"✗ Network error (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay)
            except Exception as e:
                print(f"✗ Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay)
        
        print(f"✗ Job timed out after {max_retries} attempts")
        return None
    
    def humanize_text(self, text: str, iterations: int = 1) -> str:
        """
        Humanize text through multiple iterations
        """
        current_text = text
        
        for i in range(iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {i + 1}/{iterations}")
            print(f"{'='*60}")
            print(f"Processing {len(current_text)} characters...")
            
            # Create job
            job_id = self.create_job(current_text)
            if not job_id:
                print(f"✗ Failed to create job for iteration {i + 1}")
                return current_text
            
            # Wait before polling - API needs time to process
            print(f"  Waiting 10 seconds before checking results...")
            time.sleep(10)
            
            # Get result
            humanized = self.get_job_result(job_id)
            if humanized:
                current_text = humanized
                print(f"✓ Iteration {i + 1} complete: {len(current_text)} characters")
            else:
                print(f"✗ Failed to get result for iteration {i + 1}")
                return current_text
            
            # Wait between iterations
            if i < iterations - 1:
                print(f"  Waiting 5 seconds before next iteration...")
                time.sleep(5)
        
        return current_text
    
    def process_section(self, section: Dict[str, Any], iterations: int = 2) -> Dict[str, Any]:
        """
        Process a single section, humanizing its content
        """
        processed_section = section.copy()
        
        if 'content' in section and section['content']:
            print(f"\nProcessing section: {section.get('heading', 'Untitled')}")
            processed_section['content'] = self.humanize_text(
                section['content'], 
                iterations=iterations
            )
        
        return processed_section
    
    def process_paper(self, paper_content: Dict[str, Any], 
                      iterations: int = 2) -> Dict[str, Any]:
        """
        Process entire paper through humanization
        """
        print("\n" + "="*60)
        print("STARTING PAPER HUMANIZATION")
        print("="*60)
        
        processed = {
            'title': paper_content['title'],
            'author': paper_content['author'],
            'keywords': paper_content['keywords'],
            'bibliography': paper_content['bibliography'],
            'sections': []
        }
        
        # Process abstract
        print("\n" + "-"*60)
        print("PROCESSING ABSTRACT")
        print("-"*60)
        processed['abstract'] = self.humanize_text(
            paper_content['abstract'], 
            iterations=iterations
        )
        
        # Process each section
        total_sections = len(paper_content['sections'])
        for idx, section in enumerate(paper_content['sections'], 1):
            print("\n" + "-"*60)
            print(f"PROCESSING SECTION {idx}/{total_sections}")
            print("-"*60)
            
            processed_section = self.process_section(section, iterations=iterations)
            processed['sections'].append(processed_section)
        
        print("\n" + "="*60)
        print("PAPER HUMANIZATION COMPLETE")
        print("="*60)
        
        return processed


def generate_output_file(processed_content: Dict[str, Any], 
                         output_file: str = "paper_content_humanized.py"):
    """
    Generate Python file with humanized content in original format
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('"""\n')
        f.write('Paper Content Module - Humanized Version\n')
        f.write('Processed through Decopy.ai humanizer for 0% AI detection\n')
        f.write('"""\n\n')
        
        f.write('PAPER_CONTENT = {\n')
        f.write(f'    "title": {repr(processed_content["title"])},\n\n')
        f.write(f'    "author": r"""{processed_content["author"]}""",\n\n')
        f.write(f'    "abstract": r"""{processed_content["abstract"]}""",\n\n')
        f.write(f'    "keywords": {repr(processed_content["keywords"])},\n\n')
        
        f.write('    "sections": [\n')
        
        for section in processed_content['sections']:
            f.write('        {\n')
            f.write(f'            "type": {repr(section["type"])},\n')
            f.write(f'            "heading": {repr(section["heading"])},\n')
            f.write(f'            "content": r"""{section["content"]}"""\n')
            f.write('        },\n')
        
        f.write('    ],\n\n')
        f.write(f'    "bibliography": r"""{processed_content["bibliography"]}"""\n')
        f.write('}\n')
    
    print(f"\n✓ Output saved to: {output_file}")


def main():
    # Import the original paper content
    from paper_content import PAPER_CONTENT
    
    # Initialize humanizer
    humanizer = PaperHumanizer()
    
    # Configuration
    ITERATIONS = 3  # Number of humanization passes per section
    
    print("\n" + "="*60)
    print("ACADEMIC PAPER HUMANIZER")
    print("="*60)
    print(f"Iterations per section: {ITERATIONS}")
    print(f"Total sections to process: {len(PAPER_CONTENT['sections']) + 1}")  # +1 for abstract
    print("="*60)
    
    # Process the paper
    processed_paper = humanizer.process_paper(PAPER_CONTENT, iterations=ITERATIONS)
    
    # Generate output file
    generate_output_file(processed_paper, "paper_content_humanized.py")
    
    print("\n" + "="*60)
    print("ALL PROCESSING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review paper_content_humanized.py")
    print("2. Test with AI detection tools")
    print("3. Adjust ITERATIONS if needed and re-run")


if __name__ == "__main__":
    main()
