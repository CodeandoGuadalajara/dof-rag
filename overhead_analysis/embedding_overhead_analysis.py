#!/usr/bin/env python3
"""
DuckDB Overhead Analysis for Embedding Databases
Calculates precise storage metrics and long-term projections
"""
import duckdb
import json
import os
from datetime import datetime

def format_bytes(bytes_value):
    """Convert bytes to human readable format"""
    if bytes_value < 1024:
        return f"{bytes_value:.2f} B"
    elif bytes_value < 1024**2:
        return f"{bytes_value / 1024:.2f} KB"
    elif bytes_value < 1024**3:
        return f"{bytes_value / (1024**2):.2f} MB"
    elif bytes_value < 1024**4:
        return f"{bytes_value / (1024**3):.2f} GB"
    else:
        return f"{bytes_value / (1024**4):.2f} TB"

def analyze_single_database(db_path, expected_dimension):
    """Analyze individual database and return complete metrics"""
    
    if not os.path.exists(db_path):
        print(f"❌ File not found: {db_path}")
        return None
    
    file_size = os.path.getsize(db_path)
    
    try:
        conn = duckdb.connect(db_path)
        
        # Get table information
        tables_info = conn.execute("SHOW TABLES").fetchall()
        
        # Analyze chunks table (main table)
        chunks_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        documents_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        
        # Get real embedding dimension
        if chunks_count > 0:
            embedding_sample = conn.execute("SELECT embedding FROM chunks LIMIT 1").fetchone()
            real_dimension = len(embedding_sample[0]) if embedding_sample and embedding_sample[0] else expected_dimension
        else:
            real_dimension = expected_dimension
        
        # Calculate theoretical and real sizes
        theoretical_embedding_size = real_dimension * 4  # float32 = 4 bytes
        
        # Estimated base chunk size (without embedding)
        # Based on schema: id(4) + document_id(4) + text(5000) + header(100) + page_number(4) + created_at(8)
        base_chunk_size = 4 + 4 + 5000 + 100 + 4 + 8  # 5120 bytes
        theoretical_chunk_size = base_chunk_size + theoretical_embedding_size
        
        # Theoretical total size for chunks
        theoretical_chunks_size = theoretical_chunk_size * chunks_count
        
        # Estimated size for documents (22 documents)
        # id(4) + title(~50) + url(~200) + file_path(~150) + created_at(8) = ~412 bytes
        estimated_document_size = 412
        theoretical_documents_size = estimated_document_size * documents_count
        
        # Total theoretical database size
        theoretical_total_size = theoretical_chunks_size + theoretical_documents_size
        
        # Calculate real overhead
        real_overhead = file_size - theoretical_total_size
        overhead_percentage = (real_overhead / theoretical_total_size) * 100 if theoretical_total_size > 0 else 0
        
        # Calculate per-record metrics
        size_per_chunk_real = file_size / chunks_count if chunks_count > 0 else 0
        size_per_chunk_theoretical = theoretical_chunk_size
        overhead_per_chunk = size_per_chunk_real - size_per_chunk_theoretical
        
        conn.close()
        
        return {
            'dimension': real_dimension,
            'file_path': db_path,
            'file_size_bytes': file_size,
            'file_size_mb': file_size / (1024**2),
            'chunks_count': chunks_count,
            'documents_count': documents_count,
            'theoretical_embedding_size': theoretical_embedding_size,
            'base_chunk_size': base_chunk_size,
            'theoretical_chunk_size': theoretical_chunk_size,
            'theoretical_total_size': theoretical_total_size,
            'real_overhead_bytes': real_overhead,
            'overhead_percentage': overhead_percentage,
            'size_per_chunk_real_bytes': size_per_chunk_real,
            'size_per_chunk_theoretical_bytes': size_per_chunk_theoretical,
            'overhead_per_chunk_bytes': overhead_per_chunk,
            'size_per_chunk_real_kb': size_per_chunk_real / 1024,
            'overhead_per_chunk_kb': overhead_per_chunk / 1024
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {db_path}: {e}")
        return None

def calculate_long_term_projections(analyses):
    """Calculate projections for 25 years of storage"""
    
    # Scenario: 1 document/day for 25 years
    docs_per_day = 1
    days_per_year = 365
    years = 25
    chunks_per_doc = 300  # Based on previous analysis
    images_per_doc = 15   # Estimated
    
    total_days = days_per_year * years
    total_documents = docs_per_day * total_days
    total_chunks = total_documents * chunks_per_doc
    total_images = total_documents * images_per_doc
    
    projections = {}
    
    for analysis in analyses:
        if not analysis:
            continue
            
        dimension = analysis['dimension']
        size_per_chunk_kb = analysis['size_per_chunk_real_kb']
        
        # Chunk projections
        total_chunks_size_mb = (total_chunks * size_per_chunk_kb) / 1024
        total_chunks_size_gb = total_chunks_size_mb / 1024
        
        # Document projections (estimated size: 0.4 KB per document)
        total_documents_size_kb = total_documents * 0.4
        total_documents_size_mb = total_documents_size_kb / 1024
        
        # Image projections (estimated: 2 KB per description)
        total_images_size_kb = total_images * 2
        total_images_size_mb = total_images_size_kb / 1024
        
        # General total
        total_size_mb = total_chunks_size_mb + total_documents_size_mb + total_images_size_mb
        total_size_gb = total_size_mb / 1024
        
        projections[f"{dimension}d"] = {
            'dimension': dimension,
            'size_per_chunk_kb': size_per_chunk_kb,
            'overhead_per_chunk_kb': analysis['overhead_per_chunk_kb'],
            'total_chunks': total_chunks,
            'total_documents': total_documents,
            'total_images': total_images,
            'chunks_size_mb': total_chunks_size_mb,
            'documents_size_mb': total_documents_size_mb,
            'images_size_mb': total_images_size_mb,
            'total_size_mb': total_size_mb,
            'total_size_gb': total_size_gb
        }
    
    return {
        'scenario': {
            'docs_per_day': docs_per_day,
            'days_per_year': days_per_year,
            'years': years,
            'chunks_per_doc': chunks_per_doc,
            'images_per_doc': images_per_doc
        },
        'totals': {
            'total_days': total_days,
            'total_documents': total_documents,
            'total_chunks': total_chunks,
            'total_images': total_images
        },
        'projections': projections
    }

def print_analysis_results(analyses):
    """Print analysis results"""
    
    print("\n" + "="*80)
    print("DUCKDB OVERHEAD ANALYSIS - EMBEDDING DATABASES")
    print("="*80)
    
    print("\n📊 ANALYSIS RESULTS:")
    print("-" * 80)
    print(f"{'Dimension':<12} {'DB Size':<12} {'Chunks':<8} {'Real/Chunk':<12} {'Overhead/Chunk':<15} {'Overhead %':<12}")
    print("-" * 80)
    
    for analysis in analyses:
        if analysis:
            dimension = f"{analysis['dimension']}d"
            db_size = f"{analysis['file_size_mb']:.1f} MB"
            chunks = f"{analysis['chunks_count']:,}"
            real_per_chunk = f"{analysis['size_per_chunk_real_kb']:.2f} KB"
            overhead_per_chunk = f"{analysis['overhead_per_chunk_kb']:.2f} KB"
            overhead_pct = f"{analysis['overhead_percentage']:.1f}%"
            
            print(f"{dimension:<12} {db_size:<12} {chunks:<8} {real_per_chunk:<12} {overhead_per_chunk:<15} {overhead_pct:<12}")
    
    print("\n📋 TECHNICAL DETAILS:")
    print("-" * 60)
    
    for analysis in analyses:
        if analysis:
            print(f"\n🔍 {analysis['dimension']}d:")
            print(f"   Theoretical embedding size: {analysis['theoretical_embedding_size']:,} bytes")
            print(f"   Base chunk size: {analysis['base_chunk_size']:,} bytes") 
            print(f"   Theoretical total/chunk: {analysis['theoretical_chunk_size']:,} bytes")
            print(f"   Real size/chunk: {analysis['size_per_chunk_real_bytes']:,.0f} bytes")
            print(f"   Real overhead/chunk: {analysis['overhead_per_chunk_bytes']:,.0f} bytes")

def print_projections(projection_data):
    """Print long-term projections"""
    
    scenario = projection_data['scenario']
    totals = projection_data['totals']
    projections = projection_data['projections']
    
    print(f"\n" + "="*80)
    print("STORAGE PROJECTIONS - 25 YEARS")
    print("="*80)
    
    print(f"\n📈 SCENARIO:")
    print(f"   • {scenario['docs_per_day']} document per day")
    print(f"   • {scenario['chunks_per_doc']} chunks per document")
    print(f"   • {scenario['images_per_doc']} images per document")
    print(f"   • {scenario['years']} years of operation")
    
    print(f"\n📊 PROJECTED TOTALS:")
    print(f"   • Total documents: {totals['total_documents']:,}")
    print(f"   • Total chunks: {totals['total_chunks']:,}")
    print(f"   • Total images: {totals['total_images']:,}")
    
    print(f"\n🗄️  PROJECTIONS BY DIMENSION:")
    print("-" * 80)
    print(f"{'Dimension':<12} {'Chunks (GB)':<12} {'Docs (MB)':<12} {'Imgs (MB)':<12} {'Total (GB)':<12}")
    print("-" * 80)
    
    for dim_key in sorted(projections.keys(), key=lambda x: int(x.replace('d', ''))):
        proj = projections[dim_key]
        chunks_gb = f"{proj['chunks_size_mb'] / 1024:.2f}"
        docs_mb = f"{proj['documents_size_mb']:.2f}"
        imgs_mb = f"{proj['images_size_mb']:.2f}"
        total_gb = f"{proj['total_size_gb']:.2f}"
        
        print(f"{dim_key:<12} {chunks_gb:<12} {docs_mb:<12} {imgs_mb:<12} {total_gb:<12}")

def main():
    """Main function"""
    
    # Embedding databases to analyze
    databases = [
        ('dof_db/db_qwen_512.duckdb', 512),
        ('dof_db/db_qwen_768.duckdb', 768),
        ('dof_db/db_qwen_1024.duckdb', 1024)
    ]
    
    print("Starting DuckDB overhead analysis for embedding databases...")
    
    # Analyze each database
    analyses = []
    for db_path, expected_dimension in databases:
        print(f"Analyzing {db_path}...")
        analysis = analyze_single_database(db_path, expected_dimension)
        if analysis:
            analyses.append(analysis)
    
    if not analyses:
        print("❌ Could not analyze databases")
        return
    
    # Show analysis results
    print_analysis_results(analyses)
    
    # Calculate long-term projections
    projection_data = calculate_long_term_projections(analyses)
    print_projections(projection_data)
    
    # Save results to JSON file
    results = {
        'timestamp': datetime.now().isoformat(),
        'analyses': analyses,
        'projections': projection_data
    }
    
    output_file = 'embedding_overhead_analysis_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Analysis completed. Results saved to: {output_file}")
    
    # Show executive summary
    print(f"\n" + "="*80)
    print("EXECUTIVE SUMMARY")
    print("="*80)
    print("This analysis shows that:")
    
    if len(analyses) >= 2:
        min_dimension = min(a['dimension'] for a in analyses)
        max_dimension = max(a['dimension'] for a in analyses)
        min_analysis = next(a for a in analyses if a['dimension'] == min_dimension)
        max_analysis = next(a for a in analyses if a['dimension'] == max_dimension)
        
        size_diff = max_analysis['size_per_chunk_real_kb'] - min_analysis['size_per_chunk_real_kb']
        overhead_diff = max_analysis['overhead_per_chunk_kb'] - min_analysis['overhead_per_chunk_kb']
        
        print(f"• Changing from {min_dimension}d to {max_dimension}d increases {size_diff:.2f} KB per chunk")
        print(f"• Overhead varies by {overhead_diff:.2f} KB between dimensions")
        
        for proj_key in sorted(projection_data['projections'].keys(), key=lambda x: int(x.replace('d', ''))):
            proj = projection_data['projections'][proj_key]
            print(f"• {proj_key}: {proj['total_size_gb']:.2f} GB for 25 years of storage")

if __name__ == "__main__":
    main()