#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to replace image references in Markdown files with content
from their associated TXT files.

This script looks for patterns like ![](_page_0_Picture_0.jpeg) in MD files and replaces them
with the content of the corresponding TXT file (_page_0_Picture_0.txt) if it exists.
"""

import os
import re
import sys
from pathlib import Path


def find_md_files(root_dir):
    """
    Finds all Markdown files in the root directory and subdirectories.
    
    Args:
        root_dir (str): Root directory where to search for MD files.
        
    Returns:
        list: List of paths to Markdown files found.
    """
    md_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.md'):
                md_files.append(os.path.join(root, file))
    return md_files


def process_markdown_file(md_file_path):
    """
    Processes a Markdown file to replace image references with TXT content.
    
    Args:
        md_file_path (str): Path to the Markdown file to process.
        
    Returns:
        tuple: (bool, int) - Success of the operation and number of replacements made.
    """
    # Get the directory of the MD file
    md_dir = os.path.dirname(md_file_path)
    
    try:
        # Read the content of the MD file
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern to find image references: ![](_page_X_Picture_Y.jpeg)
        pattern = r'!\[\]\(([^)]+\.jpeg)\)'
        
        # Find all matches
        matches = re.finditer(pattern, content)
        
        replacements = 0
        modified_content = content
        
        # Process each match
        for match in matches:
            img_ref = match.group(1)  # Get the image reference (e.g., _page_0_Picture_0.jpeg)
            img_path = os.path.join(md_dir, img_ref)
            
            # Build the path to the corresponding TXT file
            txt_file = os.path.splitext(img_path)[0] + '.txt'
            
            # Check if the TXT file exists
            if os.path.exists(txt_file):
                try:
                    # Read the content of the TXT file
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        txt_content = f.read().strip()
                    
                    if txt_content:  # If the TXT file has content
                        # Replace the image reference with the TXT content
                        # We use a blockquote format with a clear heading to keep the descriptive text separate
                        # and make it more explicit that it is an image description
                        replacement = f'> **Image:** {txt_content}'
                        
                        # Make the replacement in the content
                        full_match = match.group(0)  # The complete match: ![](_page_X_Picture_Y.jpeg)
                        modified_content = modified_content.replace(full_match, replacement, 1)
                        replacements += 1
                except Exception as e:
                    print(f"Error reading TXT file {txt_file}: {e}")
        
        # If replacements were made, write the modified content
        if replacements > 0:
            with open(md_file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            print(f"✓ {md_file_path}: {replacements} image references replaced.")
            return True, replacements
        else:
            print(f"✓ {md_file_path}: No image references with associated TXT files found.")
            return True, 0
            
    except Exception as e:
        print(f"✗ Error processing {md_file_path}: {e}")
        return False, 0


def main():
    """
    Main function that executes the script.
    """
    # Check arguments
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        # If no directory is provided, use the 'dof_markdown' directory in the current directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(script_dir, 'dof_markdown')
    
    # Check that the directory exists
    if not os.path.isdir(root_dir):
        print(f"Error: Directory {root_dir} does not exist.")
        sys.exit(1)
    
    print(f"Searching for Markdown files in {root_dir}...")
    md_files = find_md_files(root_dir)
    
    if not md_files:
        print("No Markdown files found.")
        sys.exit(0)
    
    print(f"Found {len(md_files)} Markdown files.")
    
    # Statistics
    total_files = len(md_files)
    processed_files = 0
    successful_files = 0
    total_replacements = 0
    
    # Process each Markdown file
    for md_file in md_files:
        processed_files += 1
        print(f"Processing {processed_files}/{total_files}: {md_file}")
        
        success, replacements = process_markdown_file(md_file)
        if success:
            successful_files += 1
            total_replacements += replacements
    
    # Show summary
    print("\n=== Summary ===")
    print(f"Total files processed: {processed_files}")
    print(f"Files successfully processed: {successful_files}")
    print(f"Total replacements made: {total_replacements}")


if __name__ == "__main__":
    main()