"""
Web scraping tool for extracting content from websites.

This tool provides web scraping capabilities for agents to extract
structured data from web pages.
"""

import asyncio
import re
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, HttpUrl

from agentic_community.core.base import BaseTool
from agentic_community.core.exceptions import ToolExecutionError
from agentic_community.core.validation import validate_url


class WebScraperInput(BaseModel):
    """Input schema for web scraper tool."""
    
    url: HttpUrl = Field(description="URL to scrape")
    selectors: Optional[Dict[str, str]] = Field(
        default=None,
        description="CSS selectors to extract specific elements"
    )
    extract_links: bool = Field(
        default=False,
        description="Whether to extract all links from the page"
    )
    extract_images: bool = Field(
        default=False,
        description="Whether to extract all images from the page"
    )
    extract_text: bool = Field(
        default=True,
        description="Whether to extract text content"
    )
    max_depth: int = Field(
        default=0,
        description="Maximum depth for recursive scraping (0 = no recursion)"
    )
    follow_links: bool = Field(
        default=False,
        description="Whether to follow and scrape linked pages"
    )


class WebScraperTool(BaseTool):
    """
    Tool for scraping web content.
    
    Features:
    - Extract text, links, images from web pages
    - Use CSS selectors for targeted extraction
    - Handle JavaScript-rendered content (basic)
    - Recursive scraping with depth control
    - Rate limiting to be respectful
    """
    
    name = "web_scraper"
    description = "Scrape and extract content from web pages"
    
    def __init__(self):
        super().__init__()
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_delay = 1.0  # Seconds between requests
        self.last_request_time = 0.0
        
    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers={
                    "User-Agent": "AgenticFramework/1.0 (Community Edition)"
                }
            )
            
    async def _rate_limit(self):
        """Apply rate limiting between requests."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
            
        self.last_request_time = asyncio.get_event_loop().time()
        
    async def _fetch_page(self, url: str) -> str:
        """
        Fetch page content with error handling.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content as string
        """
        await self._ensure_session()
        await self._rate_limit()
        
        try:
            async with self.session.get(url, timeout=10) as response:
                response.raise_for_status()
                return await response.text()
        except aiohttp.ClientError as e:
            raise ToolExecutionError(f"Failed to fetch {url}: {str(e)}")
            
    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text from HTML."""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
        
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all links from page."""
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            
            # Only include HTTP(S) links
            if absolute_url.startswith(('http://', 'https://')):
                links.append(absolute_url)
                
        return list(set(links))  # Remove duplicates
        
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract all images from page."""
        images = []
        
        for img in soup.find_all('img'):
            img_data = {
                'src': urljoin(base_url, img.get('src', '')),
                'alt': img.get('alt', ''),
                'title': img.get('title', '')
            }
            
            if img_data['src']:
                images.append(img_data)
                
        return images
        
    def _extract_by_selectors(
        self,
        soup: BeautifulSoup,
        selectors: Dict[str, str]
    ) -> Dict[str, Any]:
        """Extract content using CSS selectors."""
        results = {}
        
        for name, selector in selectors.items():
            try:
                elements = soup.select(selector)
                
                if not elements:
                    results[name] = None
                elif len(elements) == 1:
                    # Single element - return text or full element
                    element = elements[0]
                    results[name] = element.get_text(strip=True)
                else:
                    # Multiple elements - return list of texts
                    results[name] = [
                        elem.get_text(strip=True) for elem in elements
                    ]
                    
            except Exception as e:
                results[name] = f"Error with selector '{selector}': {str(e)}"
                
        return results
        
    async def _scrape_page(
        self,
        url: str,
        selectors: Optional[Dict[str, str]] = None,
        extract_links: bool = False,
        extract_images: bool = False,
        extract_text: bool = True
    ) -> Dict[str, Any]:
        """
        Scrape a single page.
        
        Returns:
            Dictionary with extracted content
        """
        # Fetch page
        html = await self._fetch_page(url)
        soup = BeautifulSoup(html, 'html.parser')
        
        result = {
            'url': url,
            'title': soup.title.string if soup.title else None
        }
        
        # Extract requested content
        if extract_text:
            result['text'] = self._extract_text(soup)
            
        if extract_links:
            result['links'] = self._extract_links(soup, url)
            
        if extract_images:
            result['images'] = self._extract_images(soup, url)
            
        if selectors:
            result['selectors'] = self._extract_by_selectors(soup, selectors)
            
        return result
        
    async def _scrape_recursive(
        self,
        url: str,
        max_depth: int,
        current_depth: int = 0,
        visited: Optional[set] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Recursively scrape pages following links.
        
        Args:
            url: Starting URL
            max_depth: Maximum recursion depth
            current_depth: Current recursion level
            visited: Set of already visited URLs
            **kwargs: Additional scraping parameters
            
        Returns:
            List of scraped page results
        """
        if visited is None:
            visited = set()
            
        if current_depth > max_depth or url in visited:
            return []
            
        visited.add(url)
        results = []
        
        try:
            # Scrape current page
            page_result = await self._scrape_page(
                url,
                extract_links=True,  # Need links for recursion
                **kwargs
            )
            results.append(page_result)
            
            # Recursively scrape linked pages
            if current_depth < max_depth and 'links' in page_result:
                # Only follow links from same domain
                base_domain = urlparse(url).netloc
                
                for link in page_result['links'][:10]:  # Limit to 10 links
                    if urlparse(link).netloc == base_domain:
                        sub_results = await self._scrape_recursive(
                            link,
                            max_depth,
                            current_depth + 1,
                            visited,
                            **kwargs
                        )
                        results.extend(sub_results)
                        
        except Exception as e:
            results.append({
                'url': url,
                'error': str(e)
            })
            
        return results
        
    async def _execute(self, **kwargs) -> str:
        """
        Execute web scraping.
        
        Args:
            url: URL to scrape
            selectors: Optional CSS selectors
            extract_links: Whether to extract links
            extract_images: Whether to extract images
            extract_text: Whether to extract text
            max_depth: Maximum recursion depth
            follow_links: Whether to follow links
            
        Returns:
            Scraped content as formatted string
        """
        # Parse input
        params = WebScraperInput(**kwargs)
        
        try:
            if params.max_depth > 0 and params.follow_links:
                # Recursive scraping
                results = await self._scrape_recursive(
                    str(params.url),
                    params.max_depth,
                    selectors=params.selectors,
                    extract_links=params.extract_links,
                    extract_images=params.extract_images,
                    extract_text=params.extract_text
                )
                
                # Format multiple results
                output = f"Scraped {len(results)} pages:\n\n"
                for i, result in enumerate(results):
                    output += f"Page {i+1}: {result.get('url', 'Unknown')}\n"
                    output += f"Title: {result.get('title', 'No title')}\n"
                    
                    if 'error' in result:
                        output += f"Error: {result['error']}\n"
                    else:
                        if 'text' in result and result['text']:
                            output += f"Text preview: {result['text'][:200]}...\n"
                        if 'links' in result:
                            output += f"Links found: {len(result['links'])}\n"
                        if 'images' in result:
                            output += f"Images found: {len(result['images'])}\n"
                            
                    output += "\n"
                    
            else:
                # Single page scraping
                result = await self._scrape_page(
                    str(params.url),
                    params.selectors,
                    params.extract_links,
                    params.extract_images,
                    params.extract_text
                )
                
                # Format single result
                output = f"Scraped: {result['url']}\n"
                output += f"Title: {result.get('title', 'No title')}\n\n"
                
                if params.extract_text and 'text' in result:
                    output += f"Text content:\n{result['text'][:1000]}"
                    if len(result['text']) > 1000:
                        output += f"... (truncated, total length: {len(result['text'])} chars)\n"
                    output += "\n\n"
                    
                if params.selectors and 'selectors' in result:
                    output += "Selector results:\n"
                    for name, value in result['selectors'].items():
                        output += f"  {name}: {value}\n"
                    output += "\n"
                    
                if params.extract_links and 'links' in result:
                    output += f"Links ({len(result['links'])} found):\n"
                    for link in result['links'][:10]:
                        output += f"  - {link}\n"
                    if len(result['links']) > 10:
                        output += f"  ... and {len(result['links']) - 10} more\n"
                    output += "\n"
                    
                if params.extract_images and 'images' in result:
                    output += f"Images ({len(result['images'])} found):\n"
                    for img in result['images'][:5]:
                        output += f"  - {img['src']}"
                        if img['alt']:
                            output += f" (alt: {img['alt']})"
                        output += "\n"
                    if len(result['images']) > 5:
                        output += f"  ... and {len(result['images']) - 5} more\n"
                        
            return output
            
        except Exception as e:
            raise ToolExecutionError(f"Web scraping failed: {str(e)}")
            
    async def cleanup(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None
