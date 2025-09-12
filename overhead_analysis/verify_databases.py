#!/usr/bin/env python3
"""
Script to verify embedding databases before overhead analysis
"""
import duckdb
import os

def verify_database(db_path, expected_dimension):
    """Verify database structure and content"""
    print(f"\n=== VERIFYING: {db_path} (Expected dimension: {expected_dimension}d) ===")
    
    if not os.path.exists(db_path):
        print(f"❌ File not found: {db_path}")
        return None
    
    # File size
    file_size = os.path.getsize(db_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"📁 File size: {file_size_mb:.2f} MB")
    
    try:
        conn = duckdb.connect(db_path)
        
        # Get table list
        tables = conn.execute("SHOW TABLES").fetchall()
        print(f"📊 Tables found: {[t[0] for t in tables]}")
        
        results = {}
        
        for table in tables:
            table_name = table[0]
            print(f"\n--- Table: {table_name} ---")
            
            # Get table schema
            schema = conn.execute(f"DESCRIBE {table_name}").fetchall()
            print("Columns:")
            for col in schema:
                print(f"  - {col[0]}: {col[1]}")
            
            # Count records
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"📈 Total records: {count}")
            
            # If it's the chunks table, verify embedding dimension
            if table_name == 'chunks' and count > 0:
                try:
                    # Try to get embedding dimension
                    embedding_sample = conn.execute(f"SELECT embedding FROM {table_name} LIMIT 1").fetchone()
                    if embedding_sample and embedding_sample[0]:
                        real_dimension = len(embedding_sample[0])
                        print(f"🎯 Real embedding dimension: {real_dimension}d")
                        
                        if real_dimension == expected_dimension:
                            print("✅ Dimension matches expected")
                        else:
                            print(f"⚠️  Dimension mismatch! Expected: {expected_dimension}d, Real: {real_dimension}d")
                    else:
                        print("⚠️  Could not get embedding sample")
                except Exception as e:
                    print(f"⚠️  Error verifying dimension: {e}")
            
            results[table_name] = {
                'count': count,
                'columns': [col[0] for col in schema]
            }
        
        conn.close()
        
        return {
            'file_size_mb': file_size_mb,
            'file_size_bytes': file_size,
            'tables': results
        }
        
    except Exception as e:
        print(f"❌ Error analyzing database: {e}")
        return None

def main():
    # Databases to verify
    databases = [
        ('dof_db/db_qwen_512.duckdb', 512),
        ('dof_db/db_qwen_768.duckdb', 768),
        ('dof_db/db_qwen_1024.duckdb', 1024)
    ]
    
    print("="*80)
    print("EMBEDDING DATABASES VERIFICATION")
    print("="*80)
    
    results = {}
    
    for db_path, dimension in databases:
        result = verify_database(db_path, dimension)
        if result:
            results[f"{dimension}d"] = result
    
    print(f"\n{'='*80}")
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    if results:
        print("\n📋 Verified databases:")
        for dim, data in results.items():
            chunks_count = data['tables'].get('chunks', {}).get('count', 0)
            print(f"  - {dim}: {data['file_size_mb']:.2f} MB, {chunks_count} chunks")
        
        print("\n✅ All databases ready for overhead analysis")
    else:
        print("\n❌ Could not verify databases")

if __name__ == "__main__":
    main()