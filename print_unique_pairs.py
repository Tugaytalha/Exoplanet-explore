import pandas as pd

# Read the CSV file
df = pd.read_csv('data/koi_with_relative_location.csv')

# Get unique kepid and kepoi_name pairs
unique_pairs = df[['kepid', 'kepoi_name']].drop_duplicates()

# Print the unique pairs
print(f"Total unique kepid, kepoi_name pairs: {len(unique_pairs)}\n")
print("kepid, kepoi_name pairs:")
print("-" * 40)

for index, row in unique_pairs.iterrows():
    print(f"{row['kepid']}, {row['kepoi_name']}")


print(f"Total unique kepid, kepoi_name pairs: {len(unique_pairs)}")
