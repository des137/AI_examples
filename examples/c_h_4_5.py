"""
Quantum Computing Financial Website Analyzer

This module fetches website content and uses OpenAI's API to analyze it
from a quantum computing financial applications perspective.
"""

import os
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import Markdown, display

from scraper import fetch_website_contents


class QuantumFinanceAnalyzer:
    """Analyzes websites for quantum computing applications in finance."""
    
    SYSTEM_PROMPT = """
    You are a quantum computing financial assistant that analyzes the contents of a website,
    and provides a short, financially relevant set of topics for quantum computing applications 
    in financial markets, ignoring text that might be navigation related.
    Think about these papers 
    1. https://www.nature.com/articles/s42254-023-00603-1 
    2. https://www.sciencedirect.com/science/article/pii/S2405428318300571
    3. https://ieeexplore.ieee.org/abstract/document/9222275
    and add insights from them.
    Respond in markdown. Do not wrap the markdown in a code block - respond just with the markdown.
    """
    
    USER_PROMPT_PREFIX = """
    Here are the contents of a website.
    Provide a short summary of this website.
    If it includes news or announcements, then summarize these too.
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-4.1-nano"):
        """
        Initialize the analyzer.
        
        Args:
            api_key: OpenAI API key. If None, loads from environment.
            model: OpenAI model to use for analysis.
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
    
    def _build_messages(self, website_content: str) -> List[Dict[str, str]]:
        """Build the message list for the API call."""
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.USER_PROMPT_PREFIX + website_content}
        ]
    
    def summarize(self, url: str) -> str:
        """
        Fetch and summarize a website from a quantum finance perspective.
        
        Args:
            url: The URL to analyze.
            
        Returns:
            Markdown-formatted summary.
        """
        website_content = fetch_website_contents(url)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(website_content)
        )
        
        return response.choices[0].message.content
    
    def display_summary(self, url: str) -> None:
        """
        Fetch, summarize, and display a website analysis.
        
        Args:
            url: The URL to analyze.
        """
        summary = self.summarize(url)
        display(Markdown(summary))


def main():
    """Main entry point for the script."""
    load_dotenv(override=True)
    
    analyzer = QuantumFinanceAnalyzer()
    analyzer.display_summary("https://finance.yahoo.com")


if __name__ == "__main__":
    main()
