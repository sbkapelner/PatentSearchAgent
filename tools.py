from langchain.tools import Tool
from firecrawl import FirecrawlApp
import os

# Initialize Firecrawl API
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

def scrape_patents_freepatentsonline(query: str, pages: int = 2, score_threshold: int = 600) -> str:
    """Scrape patents from FreePatentsOnline using Firecrawl API."""
    base_url = "https://www.freepatentsonline.com/result.html"
    extracted_patents = []

    # First handle JSON format if present
    if isinstance(query, str) and query.startswith('{"__arg1":'):
        import json
        try:
            data = json.loads(query)
            query = data["__arg1"]
            if "__arg2" in data:
                score_threshold = int(data["__arg2"])
        except:
            pass

    # Then extract query and score_threshold from input
    if isinstance(query, str) and 'minimum score threshold:' in query:
        # Extract query and threshold from the message
        parts = query.split('minimum score threshold:')
        if len(parts) == 2:
            query = parts[0].replace('Search for patents using query:', '').strip()
            try:
                score_threshold = int(parts[1].strip())
            except ValueError:
                pass  # Keep default if conversion fails
    elif isinstance(query, dict):
        score_threshold = query.get('score_threshold', score_threshold)
        query = query.get('query', '')

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
            patents = parse_patent_markdown(markdown_text, score_threshold)
            if patents:
                extracted_patents.extend(patents)
                
        except Exception as e:
            continue

    # Format the results
    if not extracted_patents:
        return "No patents found matching your search criteria."

    # Format the output
    result = "🔍 Patent Search Results\n"
    result += "===================\n\n"
    result += f"Query: '{query}'\n"
    result += f"Score Threshold: {score_threshold}\n"
    result += f"Found {len(extracted_patents)} matching patents\n\n"
    
    for i, patent in enumerate(extracted_patents, start=1):
        result += f"📄 Patent {i}\n"
        result += "---------\n"
        result += f"**[{patent['title']}]({patent['url']})** \n"
        result += f"Patent Number: `{patent['patent_number']}`\n"
        result += f"Relevance Score: {patent['score']}\n\n"

    return result

def parse_patent_markdown(markdown_text: str, score_threshold: int) -> list:
    """Parses markdown text and extracts patents with scores."""
    extracted_data = []
    lines = markdown_text.split('\n')
    
    # Find table header to locate score column
    score_col_idx = -1
    for i, line in enumerate(lines):
        if '| Match | Document | Document Title | Score |' in line:
            score_col_idx = 3  # Score is the 4th column (0-based index 3)
            break
    
    if score_col_idx == -1:
        return extracted_data
    
    # Process table rows
    current_score = None
    for i, line in enumerate(lines):
        try:
            if '|' not in line:
                continue
                
            # Split line into columns and clean up
            cols = [col.strip() for col in line.split('|')]
            if len(cols) < 5:  # Need at least 5 columns (including empty first/last)
                continue
                
            # Try to extract score
            try:
                current_score = int(cols[score_col_idx + 1])
            except:
                continue
                
            # Check score threshold
            if current_score < score_threshold:
                continue
                
            # Extract title and URL
            title_col = cols[3]  # Document Title column
            if '[' not in title_col or ']' not in title_col or '(' not in title_col or ')' not in title_col:
                continue
                
            title_start = title_col.find('[')
            title_end = title_col.find(']')
            url_start = title_col.find('(')
            url_end = title_col.find(')')
            
            if title_start == -1 or title_end == -1 or url_start == -1 or url_end == -1:
                continue
                
            title = title_col[title_start + 1:title_end]
            url = title_col[url_start + 1:url_end]
            
            # Skip non-patent links
            if 'result.html' in url or 'search.html' in url:
                continue
                
            # Extract patent number
            patent_num = url.split('/')[-1].replace('.html', '')
            
            # Add patent to results
            patent_info = {
                'patent_number': patent_num,
                'title': title,
                'url': url,
                'score': current_score
            }
            extracted_data.append(patent_info)
                
        except Exception as e:
            continue
            
    return extracted_data

# Define the tool for use with LangGraph
firecrawl_tool = Tool(
    name="firecrawl_patent_scraper",
    func=scrape_patents_freepatentsonline,
    description="Scrapes patent data from FreePatentsOnline for a given search term and returns all matching patents."
)
