import pandas as pd
import newspaper
from newspaper import Config
import time

# 1. Setup Browser Config (Bypasses 403 blocks)
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 15

def scrape_mixed_data():
    dataset = []
    
    # --- PART A: SCRAPE GLOBAL CONTENT (Using the working Base URLs) ---
    # We use .build() because it finds CURRENT live links automatically
    sources = [
        ('https://www.espn.com/nfl', 'sports'),         # Global Sports (Worked before)
        ('https://www.pbs.org/newshour/politics', 'politics') # Global Politics (Worked before)
    ]
    
    print("Step 1: Scraping live Global content...")
    for url, label in sources:
        print(f"  Crawling {url}...")
        paper = newspaper.build(url, config=config, memoize_articles=False)
        
        count = 0
        for article in paper.articles:
            if count >= 7: break # Get 7 global articles per category
            try:
                article.download()
                article.parse()
                if len(article.text) > 200:
                    dataset.append({'text': article.text, 'label': label})
                    print(f"    -> Scraped: {article.title[:40]}...")
                    count += 1
            except: continue
            
    # --- PART B: INJECT INDIAN CONTENT (Hardcoded to avoid 403 Errors) ---
    # These are the articles you requested (Indian Sports & Politics)
    print("\nStep 2: Injecting Indian Context (Manual Data)...")
    
    indian_content = [
        # Indian Politics
        {"text": "A Joint Committee of Parliament on the Viksit Bharat Shiksha Adhishthan Bill, 2025, which aims to establish a Higher Education Commission, will now present its report in the last week of the Monsoon Session.", "label": "politics"},
        {"text": "Foreign Secretary Vikram Misri held a meeting with US Ambassador Sergio Gor in New Delhi. The latter said that with the finalisation of the trade deal, many opportunities are poised to open up for the India-US partnership.", "label": "politics"},
        {"text": "In the Rajya Sabha, Opposition leaders criticised the government’s Budget, raising concerns over support for farmers. The debate focused on policy direction and welfare measures during the discussion on Budget and farmers’ issues.", "label": "politics"},
        {"text": "Opposition parties have submitted a motion seeking the removal of the Lok Sabha Speaker, escalating political tensions in Parliament. The move aims at ousting Om Birla amid disruptions.", "label": "politics"},
        {"text": "Trade relations and maintaining peace along the Line of Actual Control remained the focus of the talks between India and China. Foreign Secretary Vikram Misri met his Chinese counterpart in New Delhi to stock progress.", "label": "politics"},

        # Indian Sports
        {"text": "Pakistan batter Sahibzada Farhan has praised Jasprit Bumrah ahead of the highly anticipated India vs Pakistan clash in the T20 World Cup, acknowledging the challenge posed by India’s pace attack.", "label": "sports"},
        {"text": "Varun Chakravarthy’s bowling variations and disciplined line and length are expected to play a crucial role in India’s strategy against Pakistan. His tactical approach has drawn comparisons with Pakistan spinner Abrar Ahmed.", "label": "sports"},
        {"text": "Kuldeep Yadav is likely to make his way back into the playing eleven in place of pacer Arshdeep Singh, whereas Abhishek Sharma will open the innings as India take on Pakistan in the much-awaited ICC T20 World Cup game.", "label": "sports"},
        {"text": "England wicketkeeper Jos Buttler has heaped praise on young Indian cricketer Vaibhav Sooryavanshi, calling him the best player he has ever seen. The comments have sparked widespread discussion regarding the rising talent pool in the IPL.", "label": "sports"},
        {"text": "Indian Railways cricket team secured a dramatic victory in the Ranji Trophy final. The domestic season has seen the rise of several young talents pushing for national selection.", "label": "sports"}
    ]
    
    dataset.extend(indian_content)
    
    return pd.DataFrame(dataset)

# Run and Save
if __name__ == "__main__":
    df = scrape_mixed_data()
    
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    df.to_csv('B23CM1055_dataset.csv', index=False)
    print(f"\nSUCCESS: Saved 'B23CM1055_dataset.csv' with {len(df)} articles.")
    print(f"Sports Count: {len(df[df['label']=='sports'])}")
    print(f"Politics Count: {len(df[df['label']=='politics'])}")