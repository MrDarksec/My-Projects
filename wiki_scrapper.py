import requests
from bs4 import BeautifulSoup
import re
import os
import json

def get_page_content(title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": title,
        "format": "json",
        "prop": "text|links",
        "redirects": ""
    }
    headers = {
        "User-Agent": "physics_extractor/1.0 (freespirited07813@gmail.com)"
    }
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        
        if 'error' in data:
            print(f"API Error fetching {title}: {data['error']['info']}")
            return None, []
        
        if 'parse' not in data or 'text' not in data['parse'] or '*' not in data['parse']['text']:
            print(f"Unexpected API response structure for {title}")
            print(f"Response: {json.dumps(data, indent=2)}")
            return None, []
        
        html_content = data['parse']['text']['*']
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract main content
        content = extract_main_content(soup)
        
        # Extract links
        links = [link['*'] for link in data['parse'].get('links', []) if link.get('ns') == 0]
        
        return content, links
    except requests.RequestException as e:
        print(f"Request Error fetching {title}: {str(e)}")
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error for {title}: {str(e)}")
    except Exception as e:
        print(f"Unexpected Error fetching {title}: {str(e)}")
    
    return None, []

def extract_main_content(soup):
    # Remove unwanted elements
    for element in soup(['table', 'div', 'script', 'style', 'sup', 'span']):
        element.decompose()
    
    # Get all paragraphs
    paragraphs = soup.find_all('p')
    
    # Extract and clean text from paragraphs
    content = []
    for p in paragraphs:
        text = p.get_text()
        # Remove citations like [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        if text:
            content.append(text)
    
    return '\n\n'.join(content)

def save_content(title, content):
    filename = f"{title}.txt"
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Successfully saved content to {filename}")
    except Exception as e:
        print(f"Error saving content to {filename}: {str(e)}")

def main():
    main_article = "Physics"
    visited = set()
    to_visit = [main_article]
    
    for i in range(6):  # Main article + 5 related articles
        if not to_visit:
            break
        
        title = to_visit.pop(0)
        if title in visited:
            continue
        
        print(f"Fetching content for: {title}")
        content, links = get_page_content(title)
        
        if content:
            print(f"Content fetched for {title}. Length: {len(content)} characters")
            save_content(title, content)
            visited.add(title)
        else:
            print(f"No content fetched for {title}")
        
        if i == 0:  # For the main article, add links to visit
            to_visit.extend([link for link in links if link not in visited][:5])

if __name__ == "__main__":
    main()
