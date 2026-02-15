import pandas as pd

# Load both files
kaggle_df = pd.read_csv('kaggle_data.csv')
try:
    scraped_df = pd.read_csv('B23CM1055_dataset.csv')
    print(f"Merging with {len(scraped_df)} scraped articles...")
    
    # Combine and Shuffle
    final_df = pd.concat([kaggle_df, scraped_df], ignore_index=True)
    final_df = final_df.sample(frac=1).reset_index(drop=True)
    
except FileNotFoundError:
    print("Warning: 'B23CM1055_dataset.csv' not found. Using only Kaggle data.")
    final_df = kaggle_df

# Save Final Dataset
final_df.to_csv('final_dataset.csv', index=False)
print(f"Done! 'final_dataset.csv' is ready with {len(final_df)} articles.")