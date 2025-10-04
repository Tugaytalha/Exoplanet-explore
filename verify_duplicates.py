"""
Verify duplicate handling in the dataset
"""
import pandas as pd

print("="*70)
print("DUPLICATE ANALYSIS")
print("="*70)

# Load data
df = pd.read_csv('data/koi_with_relative_location.csv', low_memory=False)

print(f"\nOriginal dataset:")
print(f"  Total rows: {len(df):,}")
print(f"  Total columns: {len(df.columns)}")

# Check for duplicates
unique_pairs = df[['kepid', 'kepoi_name']].drop_duplicates()
print(f"\nUnique KOI pairs (kepid, kepoi_name):")
print(f"  Unique pairs: {len(unique_pairs):,}")
print(f"  Duplicate rows: {len(df) - len(unique_pairs):,}")
print(f"  Duplication ratio: {(len(df) / len(unique_pairs)):.2f}x")

# Show examples of duplicates
print(f"\nExamples of duplicates:")
print("-"*70)

duplicates = df[df.duplicated(subset=['kepid', 'kepoi_name'], keep=False)]
if not duplicates.empty:
    # Group by kepid and kepoi_name and show first few groups
    grouped = duplicates.groupby(['kepid', 'kepoi_name'])
    
    for i, ((kepid, kepoi_name), group) in enumerate(grouped):
        if i >= 3:  # Show only first 3 examples
            break
        print(f"\nKOI: {kepoi_name} (kepid={kepid}) - {len(group)} occurrences")
        
        # Show which columns differ between duplicates
        if len(group) > 1:
            # Check st_refname and sy_refname which likely differ
            if 'st_refname' in group.columns:
                refs = group['st_refname'].unique()
                if len(refs) > 1:
                    print(f"  Different references: {refs[:3]}")

# After deduplication
df_dedup = df.drop_duplicates(subset=['kepid', 'kepoi_name'], keep='first')

print(f"\n" + "="*70)
print("AFTER DEDUPLICATION (keep='first'):")
print("="*70)
print(f"  Rows: {len(df_dedup):,}")
print(f"  Removed: {len(df) - len(df_dedup):,} duplicate rows")
print(f"  Memory saved: ~{((len(df) - len(df_dedup)) / len(df) * 100):.1f}%")

print("\n✅ API will use only the first occurrence of each KOI")
print("   This ensures unique kepid values and faster queries")

