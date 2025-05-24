"""
Web Scraper Tool for Community Edition

Provides web scraping capabilities with respect for robots.txt
and rate limiting.
"""

import asyncio
import aiohttp
from typing import Dict, Any, List, Optional
from urllib.parse import urljoin, urlparse
import re
from datetime import datetime
import json

from agentic_community.core.base import BaseTool
from agentic_community.core.exceptions import ToolError


class WebScraperTool(BaseTool):
    """
    Tool for scraping web content.
    
    Features:
    - Async HTTP requests
    - HTML parsing
    - Link extraction
    - Text extraction
    - Metadata extraction
    - Rate limiting
    - robots.txt compliance
    """
    
    def __init__(self, 
                 rate_limit: float = 1.0,
                 timeout: int = 30,
                 user_agent: Optional[str] = None):
        super().__init__(
            name="WebScraper",
            description="Scrape and extract content from web pages"
        )
        self.rate_limit = rate_limit  # Seconds between requests
        self.timeout = timeout
        self.user_agent = user_agent or "AgenticCommunity/1.0 WebScraper"
        self._last_request_time = {}
        self._robots_cache = {}
    
    async def scrape(self, 
                    url: str,
                    extract: List[str] = None,
                    follow_links: bool = False,
                    max_depth: int = 1) -> Dict[str, Any]:
        """
        Scrape a web page and extract content.
        
        Args:
            url: URL to scrape
            extract: List of elements to extract ["text", "links", "images", "metadata"]
            follow_links: Whether to follow and scrape linked pages
            max_depth: Maximum depth for following links
            
        Returns:
            Extracted content
        """
        if extract is None:
            extract = ["text", "links", "metadata"]
        
        # Check robots.txt
        if not await self._check_robots_txt(url):
            raise ToolError(f"Scraping not allowed by robots.txt for {url}")
        
        # Rate limiting
        await self._rate_limit_check(url)
        
        # Scrape the page
        result = await self._scrape_page(url, extract)
        
        # Follow links if requested
        if follow_links and max_depth > 1 and "links" in result:
            result["linked_pages"] = await self._scrape_linked_pages(
                result["links"],
                extract,
                max_depth - 1,
                base_url=url
            )
        
        return result
    
    async def _check_robots_txt(self, url: str) -> bool:
        """
        Check if scraping is allowed by robots.txt.
        """
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        
        # Check cache
        if robots_url in self._robots_cache:
            return self._robots_cache[robots_url]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    robots_url,
                    timeout=aiohttp.ClientTimeout(total=10),
                    headers={"User-Agent": self.user_agent}
                ) as response:
                    if response.status == 200:
                        content = await response.text()
                        # Simple robots.txt parsing
                        disallow_patterns = self._parse_robots_txt(content)
                        
                        # Check if URL matches any disallow pattern
                        path = parsed.path or "/"
                        allowed = not any(
                            path.startswith(pattern) 
                            for pattern in disallow_patterns
                        )
                        
                        self._robots_cache[robots_url] = allowed
                        return allowed
                    else:
                        # No robots.txt means allowed
                        self._robots_cache[robots_url] = True
                        return True
                        
        except Exception:
            # Error checking robots.txt, assume allowed
            return True
    
    def _parse_robots_txt(self, content: str) -> List[str]:
        """
        Parse robots.txt for disallow patterns.
        """
        disallow_patterns = []
        user_agent_applies = False
        
        for line in content.split('\n'):
            line = line.strip().lower()
            
            if line.startswith('user-agent:'):
                agent = line.split(':', 1)[1].strip()
                user_agent_applies = agent == '*' or self.user_agent.lower() in agent
            
            elif line.startswith('disallow:') and user_agent_applies:
                path = line.split(':', 1)[1].strip()
                if path:
                    disallow_patterns.append(path)
        
        return disallow_patterns
    
    async def _rate_limit_check(self, url: str):
        """
        Enforce rate limiting per domain.
        """
        domain = urlparse(url).netloc
        
        if domain in self._last_request_time:
            elapsed = datetime.now().timestamp() - self._last_request_time[domain]
            if elapsed < self.rate_limit:
                await asyncio.sleep(self.rate_limit - elapsed)
        
        self._last_request_time[domain] = datetime.now().timestamp()
    
    async def _scrape_page(self, url: str, extract: List[str]) -> Dict[str, Any]:
        """
        Scrape a single page.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    headers={"User-Agent": self.user_agent}
                ) as response:
                    if response.status != 200:
                        raise ToolError(f"HTTP {response.status} for {url}")
                    
                    content = await response.text()
                    
                    result = {
                        "url": url,
                        "status": response.status,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Extract requested elements
                    if "text" in extract:
                        result["text"] = self._extract_text(content)
                    
                    if "links" in extract:
                        result["links"] = self._extract_links(content, url)
                    
                    if "images" in extract:
                        result["images"] = self._extract_images(content, url)
                    
                    if "metadata" in extract:
                        result["metadata"] = self._extract_metadata(content)
                    
                    if "html" in extract:
                        result["html"] = content
                    
                    return result
                    
        except asyncio.TimeoutError:
            raise ToolError(f"Timeout scraping {url}")
        except Exception as e:
            raise ToolError(f"Error scraping {url}: {str(e)}")
    
    def _extract_text(self, html: str) -> str:
        """
        Extract visible text from HTML.
        """
        # Remove script and style elements
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _extract_links(self, html: str, base_url: str) -> List[Dict[str, str]]:
        """
        Extract links from HTML.
        """
        links = []
        
        # Find all anchor tags
        anchor_pattern = r'<a[^>]+href=["\']([^"\'>]+)["\'][^>]*>([^<]*)</a>'
        matches = re.findall(anchor_pattern, html, re.IGNORECASE)
        
        for href, text in matches:
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            
            links.append({
                "url": absolute_url,
                "text": text.strip(),
                "relative": not href.startswith(('http://', 'https://'))
            })
        
        return links
    
    def _extract_images(self, html: str, base_url: str) -> List[Dict[str, str]]:
        """
        Extract images from HTML.
        """
        images = []
        
        # Find all img tags
        img_pattern = r'<img[^>]+src=["\']([^"\'>]+)["\'][^>]*(?:alt=["\']([^"\'>]*)["\'])?'
        matches = re.findall(img_pattern, html, re.IGNORECASE)
        
        for src, alt in matches:
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, src)
            
            images.append({
                "url": absolute_url,
                "alt": alt or "",
                "relative": not src.startswith(('http://', 'https://'))
            })
        
        return images
    
    def _extract_metadata(self, html: str) -> Dict[str, str]:
        """
        Extract metadata from HTML.
        """
        metadata = {}
        
        # Extract title
        title_match = re.search(r'<title>([^<]+)</title>', html, re.IGNORECASE)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        
        # Extract meta tags
        meta_pattern = r'<meta[^>]+(?:name|property)=["\']([^"\'>]+)["\'][^>]+content=["\']([^"\'>]+)["\']'
        meta_matches = re.findall(meta_pattern, html, re.IGNORECASE)
        
        for name, content in meta_matches:
            metadata[name] = content
        
        # Extract Open Graph tags
        og_pattern = r'<meta[^>]+property=["\']og:([^"\'>]+)["\'][^>]+content=["\']([^"\'>]+)["\']'
        og_matches = re.findall(og_pattern, html, re.IGNORECASE)
        
        for property_name, content in og_matches:
            metadata[f"og:{property_name}"] = content
        
        return metadata
    
    async def _scrape_linked_pages(self,
                                  links: List[Dict[str, str]],
                                  extract: List[str],
                                  max_depth: int,
                                  base_url: str) -> List[Dict[str, Any]]:
        """
        Scrape linked pages recursively.
        """
        results = []
        
        # Limit number of links to follow
        links_to_follow = links[:10]  # Max 10 links per level
        
        for link in links_to_follow:
            try:
                # Only follow links from same domain
                if urlparse(link["url"]).netloc == urlparse(base_url).netloc:
                    result = await self.scrape(
                        link["url"],
                        extract=extract,
                        follow_links=max_depth > 1,
                        max_depth=max_depth
                    )
                    results.append(result)
            except Exception:
                # Skip failed pages
                continue
        
        return results
    
    async def search_content(self,
                           url: str,
                           search_terms: List[str],
                           context_length: int = 100) -> List[Dict[str, Any]]:
        """
        Search for specific terms in scraped content.
        
        Args:
            url: URL to scrape and search
            search_terms: Terms to search for
            context_length: Characters of context around matches
            
        Returns:
            List of matches with context
        """
        # Scrape the page
        result = await self.scrape(url, extract=["text"])
        text = result.get("text", "")
        
        matches = []
        
        for term in search_terms:
            # Case-insensitive search
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            
            for match in pattern.finditer(text):
                start = max(0, match.start() - context_length)
                end = min(len(text), match.end() + context_length)
                
                matches.append({
                    "term": term,
                    "position": match.start(),
                    "context": text[start:end],
                    "exact_match": match.group()
                })
        
        return matches
    
    async def process(self, input_data: str) -> str:
        """
        Process scraping request.
        """
        try:
            # Parse input
            if input_data.startswith("http"):
                # Simple URL scraping
                result = await self.scrape(input_data)
            else:
                # JSON request
                request = json.loads(input_data)
                
                if "search" in request:
                    # Search mode
                    result = await self.search_content(
                        request["url"],
                        request["search"],
                        request.get("context_length", 100)
                    )
                else:
                    # Regular scraping
                    result = await self.scrape(
                        request["url"],
                        extract=request.get("extract"),
                        follow_links=request.get("follow_links", False),
                        max_depth=request.get("max_depth", 1)
                    )
            
            return json.dumps(result, indent=2)
            
        except json.JSONDecodeError:
            # Treat as URL
            result = await self.scrape(input_data)
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error: {str(e)}"
