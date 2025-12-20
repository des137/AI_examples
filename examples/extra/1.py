
import os
from dotenv import load_dotenv
from scraper import fetch_website_contents
from IPython.display import Markdown, display
from openai import OpenAI

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')
openai = OpenAI()

system_prompt = """
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

user_prompt_prefix = """
Here are the contents of a website.
Provide a short summary of this website.
If it includes news or announcements, then summarize these too.
"""

def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_prefix + website}
    ]

def summarize(url):
    website = fetch_website_contents(url)
    response = openai.chat.completions.create(
        model = "gpt-4.1-nano",
        messages = messages_for(website)
    )
    return response.choices[0].message.content

def display_summary(url):
    summary = summarize(url)
    display(Markdown(summary))

if __name__ == "__main__":
    display_summary("https://finance.yahoo.com")
