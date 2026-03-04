import requests
from urllib.parse import quote

def google_patents_first_page(query):
    
    # replicate your JS query builder
    query_plus = query.replace(" ", "+")
    qs = f"q={query_plus}&oq={query_plus}"

    # encode like encodeURIComponent
    encoded = quote(qs, safe="")

    url = f"https://patents.google.com/xhr/query?url={encoded}"

    response = requests.get(url)
    response.raise_for_status()

    return response.json()