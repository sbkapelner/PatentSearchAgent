from langchain.tools import Tool
from firecrawl import FirecrawlApp
import os

# Initialize Firecrawl API
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

def scrape_patents_freepatentsonline(query: str, pages: int = 2) -> str:
    """Scrape patents from FreePatentsOnline using Firecrawl API."""
    base_url = "https://www.freepatentsonline.com/result.html"
    extracted_patents = []

    # Extract the actual query if it's in JSON format
    if isinstance(query, str) and query.startswith('{"__arg1":'):
        import json
        try:
            query = json.loads(query)["__arg1"]
        except:
            pass

    for page in range(1, pages + 1):
        # Format query properly for the URL
        url = f"{base_url}?p={page}&sort=relevance&srch=top&query_txt={query.replace(' ', '+')}&patents_us=on&patents_other=on"

        try:
            # Use basic FireCrawl configuration
            response = app.scrape_url(
                url,
                params={"formats": ["markdown"]}
            )
            
            if not response or "markdown" not in response:
                continue

            # Parse markdown response to extract relevant patent data
            markdown_text = response["markdown"]
            patents = parse_patent_markdown(markdown_text)
            if patents:
                extracted_patents.extend(patents)
                
        except Exception as e:
            continue

    # Format the results
    if not extracted_patents:
        return "No patents found matching your search criteria."

    # Sort patents by score in descending order
    sorted_patents = sorted(extracted_patents, key=lambda x: x['score'], reverse=True)
    
    # Format the output
    result = f"🔍 Found {len(sorted_patents)} relevant wine-related patents for '{query}' (scores >= 600):\n\n"
    
    for i, patent in enumerate(sorted_patents, start=1):
        result += f"📄 Patent #{i}:\n"
        result += f"   Title: **[{patent['title']}]({patent['url']})**\n"
        result += f"   Patent Number: `{patent['patent_number']}`\n"
        result += f"   Relevance Score: {patent['score']}\n\n"

    return result

def parse_patent_markdown(markdown_text: str) -> list:
    """Parses markdown text and extracts patents."""
    extracted_data = []
    lines = markdown_text.split('\n')
    
    # Process all lines looking for patent entries
    for i, line in enumerate(lines):
        try:
            # Check if line contains a patent entry (has a URL to a specific patent)
            if 'freepatentsonline.com/' in line and '.html' in line:
                # Skip navigation links
                if 'result.html' in line or 'search.html' in line:
                    continue
                    
                # Extract patent number from URL
                url_start = line.find('(https://www.freepatentsonline.com/')
                if url_start == -1:
                    continue
                    
                url_end = line.find(')', url_start)
                if url_end == -1:
                    continue
                    
                url = line[url_start + 1:url_end]
                patent_num = url.split('/')[-1].replace('.html', '')
                
                # Extract title
                title_start = line.find('[', 0, url_start)
                title_end = line.find(']', title_start)
                if title_start == -1 or title_end == -1:
                    continue
                    
                title = line[title_start + 1:title_end]
                
                # Skip non-patent links
                if not title or title in ['<', '>', 'search again', '2', '3', '4', '5']:
                    continue
                
                # Score is usually in the next few lines
                score = 1000  # Default high score if not found
                for j in range(i, min(i + 3, len(lines))):
                    if str(j+1) in lines[j]:
                        try:
                            score = 1000 - (j - i) * 100  # Decrease score based on position
                        except:
                            continue
                        break
                
                if score >= 600:
                    patent_info = {
                        'patent_number': patent_num,
                        'title': title,
                        'url': url,
                        'score': score
                    }
                    extracted_data.append(patent_info)
                
        except Exception as e:
            continue
            
    return extracted_data

# Define the tool for use with LangGraph
firecrawl_tool = Tool(
    name="firecrawl_patent_scraper",
    func=scrape_patents_freepatentsonline,
    description="Scrapes patent data from FreePatentsOnline for a given search term and returns all patents with a score of 600 or greater."
)
