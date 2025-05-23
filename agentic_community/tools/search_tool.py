"""
Search tool implementation for Agentic Framework.

This module provides search functionality using DuckDuckGo for the community edition.
No API key required, making it easy for users to get started.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus
import aiohttp
import asyncio
from bs4 import BeautifulSoup

from agentic_community.core.base import BaseTool
from agentic_community.core.exceptions import SearchError, NetworkError, ValidationError

logger = logging.getLogger(__name__)


class SearchTool(BaseTool):
    """
    Search tool that provides web search functionality.
    
    Community edition uses DuckDuckGo (no API key required).
    Enterprise edition can use other providers like Google, Bing, etc.
    """
    
    name = "search"
    description = "Search for information on the web using DuckDuckGo"
    
    def __init__(self, max_results: int = 5, timeout: int = 10):
        """
        Initialize the search tool.
        
        Args:
            max_results: Maximum number of results to return
            timeout: Timeout for search requests in seconds
        """
        super().__init__()
        self.max_results = max_results
        self.timeout = timeout
        self.session = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
        return self.session
    
    async def search_async(self, query: str) -> List[Dict[str, str]]:
        """
        Perform asynchronous web search using DuckDuckGo.
        
        Args:
            query: Search query
            
        Returns:
            List of search results with title, link, and snippet
        """
        if not query or not query.strip():
            raise ValidationError("query", query, "Search query cannot be empty")
        
        try:
            session = await self._get_session()
            
            # DuckDuckGo HTML search URL
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            
            logger.info(f"Searching DuckDuckGo for: {query}")
            
            async with session.get(url) as response:
                if response.status != 200:
                    raise NetworkError(
                        f"Search request failed with status {response.status}",
                        error_code="SEARCH_HTTP_ERROR",
                        details={"status": response.status, "url": url}
                    )
                
                html = await response.text()
                
            # Parse results
            results = self._parse_duckduckgo_results(html)
            
            # Limit results
            results = results[:self.max_results]
            
            logger.info(f"Found {len(results)} results for query: {query}")
            return results
            
        except asyncio.TimeoutError:
            raise SearchError(query, f"Search timed out after {self.timeout} seconds")
        except aiohttp.ClientError as e:
            raise NetworkError(
                f"Network error during search: {str(e)}",
                error_code="SEARCH_NETWORK_ERROR",
                details={"query": query, "error": str(e)}
            )
        except Exception as e:
            logger.exception(f"Unexpected error during search for query: {query}")
            raise SearchError(query, str(e))
    
    def _parse_duckduckgo_results(self, html: str) -> List[Dict[str, str]]:
        """
        Parse DuckDuckGo HTML results.
        
        Args:
            html: HTML content from DuckDuckGo
            
        Returns:
            List of parsed results
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            results = []
            
            # Find result divs
            result_divs = soup.find_all('div', class_='links_main')
            
            for div in result_divs:
                try:
                    # Extract title and link
                    link_elem = div.find('a', class_='result__a')
                    if not link_elem:
                        continue
                    
                    title = link_elem.get_text(strip=True)
                    link = link_elem.get('href', '')
                    
                    # Extract snippet
                    snippet_elem = div.find('a', class_='result__snippet')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    if title and link:
                        results.append({
                            'title': title,
                            'link': link,
                            'snippet': snippet
                        })
                except Exception as e:
                    logger.debug(f"Error parsing individual result: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error parsing DuckDuckGo results: {e}")
            return []
    
    def search(self, query: str) -> str:
        """
        Perform synchronous web search.
        
        Args:
            query: Search query
            
        Returns:
            Formatted string of search results
        """
        try:
            # Run async search in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(self.search_async(query))
            finally:
                loop.close()
            
            if not results:
                return f"No results found for '{query}'"
            
            # Format results
            formatted = f"Search results for '{query}':\n\n"
            for i, result in enumerate(results, 1):
                formatted += f"{i}. {result['title']}\n"
                formatted += f"   Link: {result['link']}\n"
                if result['snippet']:
                    formatted += f"   {result['snippet']}\n"
                formatted += "\n"
            
            return formatted.strip()
            
        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _invoke(self, query: str) -> str:
        """
        Tool invocation method used by agents.
        
        Args:
            query: Search query
            
        Returns:
            Search results as formatted string
        """
        return self.search(query)
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if self.session and not self.session.closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    loop.run_until_complete(self.close())
            except Exception:
                pass  # Ignore errors during cleanup


class EnhancedSearchTool(SearchTool):
    """
    Enhanced search tool with additional features (Enterprise Edition).
    
    This is a placeholder for enterprise features like:
    - Multiple search providers (Google, Bing, etc.)
    - Advanced filtering
    - Result ranking
    - Caching
    """
    
    def __init__(self, provider: str = "duckduckgo", api_key: Optional[str] = None, **kwargs):
        """
        Initialize enhanced search tool.
        
        Args:
            provider: Search provider to use
            api_key: API key for the provider (if required)
            **kwargs: Additional provider-specific arguments
        """
        super().__init__(**kwargs)
        self.provider = provider
        self.api_key = api_key
        
        # This would be implemented in enterprise edition
        if provider != "duckduckgo":
            from agentic_community.core.exceptions import FeatureNotAvailableError
            raise FeatureNotAvailableError(
                f"Search provider '{provider}'",
                edition="community",
                required_edition="enterprise"
            )


# For backward compatibility
def search_web(query: str, max_results: int = 5) -> str:
    """
    Convenience function for web search.
    
    Args:
        query: Search query
        max_results: Maximum number of results
        
    Returns:
        Formatted search results
    """
    tool = SearchTool(max_results=max_results)
    try:
        return tool.search(query)
    finally:
        # Clean up
        asyncio.run(tool.close())
