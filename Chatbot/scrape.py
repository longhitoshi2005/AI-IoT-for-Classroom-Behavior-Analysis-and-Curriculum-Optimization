import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

urls = [
    "https://www.unesco.org/en/education",
    "https://www.oecd.org/en/about/directorates/directorate-for-education-and-skills.html",
    "https://www.edutopia.org/",
    "https://www.teachthought.com/?utm_source=chatgpt.com",
    "https://www.tes.com/teaching-resources",
    "https://www.khanacademy.org/teacher-resources",
    "https://iste.org/"
]

all_text = []

def normalize_url(u: str) -> str:
    parsed = urlparse(u)
    if not parsed.scheme:
        return f"https://{u}"
    return u

session = requests.Session()
retry = Retry(
    total=3,
    connect=3,
    read=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}

for raw_url in urls:
    url = normalize_url(raw_url)
    print(f"Scraping {url}")
    try:
        resp = session.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        html = resp.text
    except RequestException as e:
        print("Request failed:", e)
        continue

    soup = BeautifulSoup(html, 'html.parser')
    for script in soup(["script", "style", "noscript"]):
        script.extract()
    text = soup.get_text(separator=' ')
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    content = ' '.join(lines)
    all_text.append(f"URL: {url}\n\n{content}")

with open("education_content.txt", "w", encoding='utf-8') as f:
    f.write("\n\n".join(all_text))

print("Content saved to education_content.txt")
