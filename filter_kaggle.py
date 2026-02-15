import pandas as pd

def process_kaggle_data():
    # 1. Load the file you just uploaded
    try:
        df = pd.read_csv('BBC_News_processed.csv')
        print("File loaded successfully.")
    except FileNotFoundError:
        print("Error: 'BBC_News_processed.csv' not found.")
        return

    # 2. Filter for only 'sport' and 'politics'
    # We ignore business, tech, and entertainment for this assignment
    filtered_df = df[df['Category'].isin(['sport', 'politics'])].copy()
    
    # 3. Rename columns to match your scraper's format
    # The scraper produces 'text' and 'label', so we match that here.
    filtered_df = filtered_df[['Text', 'Category']].rename(columns={
        'Text': 'text', 
        'Category': 'label'
    })
    
    # 4. Save to a new clean CSV
    filtered_df.to_csv('kaggle_data.csv', index=False)
    print(f"Success! Filtered {len(filtered_df)} articles.")
    print(filtered_df['label'].value_counts())

if __name__ == "__main__":
    process_kaggle_data()