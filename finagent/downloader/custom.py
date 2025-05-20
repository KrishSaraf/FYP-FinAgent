import requests
from typing import Optional, Dict, Any

class Downloader:
    """
    Base downloader class with basic request functionality.
    Simplified version focused on Indian market data downloading.
    """
    def __init__(self,
                 *args,
                 max_retry: int = 3,
                 timeout: int = 30,
                 **kwargs):
        """
        Initialize the downloader with basic request parameters.
        
        Args:
            max_retry (int): Maximum number of retry attempts for failed requests
            timeout (int): Request timeout in seconds
        """
        self.max_retry = max_retry
        self.timeout = timeout
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })

    def _request_get(self, 
                    url: str, 
                    headers: Optional[Dict[str, str]] = None, 
                    params: Optional[Dict[str, Any]] = None) -> Optional[requests.Response]:
        """
        Make a GET request with retry logic.
        
        Args:
            url (str): The URL to request
            headers (Dict[str, str], optional): Additional headers
            params (Dict[str, Any], optional): Query parameters
            
        Returns:
            Optional[requests.Response]: Response object if successful, None otherwise
        """
        if headers:
            self.session.headers.update(headers)

        for attempt in range(self.max_retry):
            try:
                response = self.session.get(
                    url=url,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retry - 1:  # Last attempt
                    raise e
                continue

    def _request_post(self, 
                     url: str, 
                     headers: Optional[Dict[str, str]] = None, 
                     json: Optional[Dict[str, Any]] = None) -> Optional[requests.Response]:
        """
        Make a POST request with retry logic.
        
        Args:
            url (str): The URL to request
            headers (Dict[str, str], optional): Additional headers
            json (Dict[str, Any], optional): JSON data to send
            
        Returns:
            Optional[requests.Response]: Response object if successful, None otherwise
        """
        if headers:
            self.session.headers.update(headers)

        for attempt in range(self.max_retry):
            try:
                response = self.session.post(
                    url=url,
                    json=json,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retry - 1:  # Last attempt
                    raise e
                continue

    def close(self):
        """Close the session when done."""
        self.session.close()