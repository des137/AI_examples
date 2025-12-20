import os
import sys
from typing import List, Dict

# Third-party imports
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from IPython.display import Markdown, display

# Local imports
# Ensure 'scraper.py' is in the same directory or python path
try:
    from scraper import fetch_website_contents
except ImportError:
    print("Error: Could not import 'scraper'. Ensure 'scraper.py' exists.")
    sys.exit(1)

# --- Configuration & Constants ---

MODEL_NAME = "gpt-4o-mini" # Updated to a standard model (4.1-nano is not currently standard)

SYSTEM_PROMPT = """
You are a quantum computing financial assistant. 
Your goal is to analyze website contents and provide a short, financially relevant 
set of topics for quantum computing applications in financial markets.

Ignore text that might be navigation-related.

Integrate insights from the following papers into your analysis:
1. https://www.nature.com/articles/s42254-023-00603-1
2. https://www.sciencedirect.com/science/article/pii/S2405428318300571
3. https://ieeexplore.ieee.org/abstract/document/9222275

Output format: Markdown (do not wrap in code blocks).
"""

USER_PROMPT_PREFIX = """
Here are the contents of a website.
Provide a short summary of this website.
If it includes news or announcements, then summarize these too.

Website Content:
"""

# --- Core Logic ---

class QuantumFinanceAnalyst:
    def __init__(self):
        load_dotenv(override=True)
        self.api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
            
        self.client = OpenAI(api_key=self.api_key)

    def _construct_messages(self, website_content: str) -> List[Dict[str, str]]:
        """Constructs the message history for the API call."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{USER_PROMPT_PREFIX}\n\n{website_content}"}
        ]

    def analyze_url(self, url: str) -> str:
        """Fetches content from a URL and generates a summary."""
        try:
            print(f"Fetching contents from: {url}...")
            website_content = fetch_website_contents(url)
            
            if not website_content:
                return "Error: No content retrieved from the website."

            print("Generating analysis...")
            messages = self._construct_messages(website_content)
            
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages
            )
            return response.choices[0].message.content

        except OpenAIError as e:
            return f"OpenAI API Error: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"

# --- Execution ---

def main():
    target_url = "https://finance.yahoo.com"
    
    try:
        analyst = QuantumFinanceAnalyst()
        summary = analyst.analyze_url(target_url)
        display(Markdown(summary))
        
    except ValueError as ve:
        print(f"Configuration Error: {ve}")

if __name__ == "__main__":
    main()
