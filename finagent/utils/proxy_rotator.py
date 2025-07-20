from dataclasses import dataclass
import random
import time
from typing import List, Optional
import logging
import requests
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProxyInfo:
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    protocol: str = "http"
    
    def to_url(self) -> str:
        if self.username and self.password:
            return f"{self.protocol}://{self.username}:{self.password}@{self.host}:{self.port}"
        return f"{self.protocol}://{self.host}:{self.port}"

class ProxyRotator:
    def __init__(self, username: str, password: str, proxy_server: str = "fiip79eu.pr.thordata.net:9999", 
                 refresh_interval: int = 300, country: str = None, city: str = None, 
                 use_sessions: bool = True):
        """
        Initialize ThorData proxy rotator.
        
        Args:
            username: Your ThorData username (without 'td-customer-' prefix)
            password: Your ThorData password
            proxy_server: ThorData proxy server (default: t.pr.thordata.net:9999)
            refresh_interval: Seconds between proxy refreshes
            country: Country code (e.g., 'US', 'GB', 'CA') for geo-targeting
            city: City name (e.g., 'houston', 'london') for geo-targeting
            use_sessions: Whether to use session control for sticky IPs
        """
        self.base_username = username
        self.password = password
        self.proxy_server = proxy_server
        self.refresh_interval = refresh_interval
        self.country = country
        self.city = city
        self.use_sessions = use_sessions
        self.proxies: List[ProxyInfo] = []
        self.last_refresh = 0
        self.current_index = 0
        self.failed_proxies = set()
        self.session_ids = []

    def _generate_session_id(self) -> str:
        """Generate a unique session ID for sticky IP sessions."""
        return str(uuid.uuid4())[:8]

    def _build_username(self, session_id: str = None) -> str:
        """Build the ThorData username string with location and session parameters."""
        username_parts = [f"td-customer-{self.base_username}"]
        
        # Add country if specified
        if self.country:
            username_parts.append(f"country-{self.country}")
        
        # Add city if specified
        if self.city:
            username_parts.append(f"city-{self.city}")
        
        # Add session ID if specified
        if session_id:
            username_parts.append(f"sessid-{session_id}")
        
        return "-".join(username_parts)

    def fetch_proxies(self) -> List[ProxyInfo]:
        """Generate ThorData proxy configurations."""
        try:
            host, port = self.proxy_server.split(":")
            proxies = []
            
            if self.use_sessions:
                # Generate multiple session-based proxies
                for i in range(5):  # Create 5 different sessions
                    session_id = self._generate_session_id()
                    self.session_ids.append(session_id)
                    
                    proxy = ProxyInfo(
                        host=host,
                        port=int(port),
                        username=self._build_username(session_id),
                        password=self.password,
                        protocol="http"
                    )
                    proxies.append(proxy)
            else:
                # Single proxy without session (rotating IP each request)
                proxy = ProxyInfo(
                    host=host,
                    port=int(port),
                    username=self._build_username(),
                    password=self.password,
                    protocol="http"
                )
                proxies.append(proxy)
            
            logger.info(f"Generated {len(proxies)} ThorData proxy configurations")
            return proxies
            
        except Exception as e:
            logger.error(f"Error creating ThorData proxies: {e}")
            return []

    def refresh_proxies(self):
        """Refresh proxy list if needed."""
        current_time = time.time()
        if current_time - self.last_refresh >= self.refresh_interval:
            logger.info("Refreshing ThorData proxy list...")
            new_proxies = self.fetch_proxies()
            if new_proxies:
                self.proxies = new_proxies
                self.failed_proxies.clear()  # Reset failed proxies on refresh
                self.current_index = 0
                logger.info(f"Updated proxy list with {len(self.proxies)} proxies")
            self.last_refresh = current_time

    def get_next_proxy(self) -> Optional[ProxyInfo]:
        """Get the next working proxy in rotation."""
        if not self.proxies:
            return None

        # Filter out failed proxies
        available_proxies = [p for i, p in enumerate(self.proxies) if i not in self.failed_proxies]

        if not available_proxies:
            # If all proxies failed, reset and try again
            logger.warning("All proxies failed, resetting failed proxy list")
            self.failed_proxies.clear()
            available_proxies = self.proxies

        if not available_proxies:
            return None

        # Get next proxy (round-robin)
        proxy = available_proxies[self.current_index % len(available_proxies)]
        self.current_index += 1

        return proxy

    def mark_proxy_failed(self, proxy: ProxyInfo):
        """Mark a proxy as failed."""
        for i, p in enumerate(self.proxies):
            if p.host == proxy.host and p.port == proxy.port and p.username == proxy.username:
                self.failed_proxies.add(i)
                logger.warning(f"Marked ThorData proxy {proxy.host}:{proxy.port} (user: {proxy.username}) as failed")
                break

    def test_proxy(self, proxy: ProxyInfo) -> bool:
        """Test if a proxy is working."""
        try:
            proxy_url = proxy.to_url()
            response = requests.get(
                "https://ipinfo.thordata.com",
                proxies={"http": proxy_url, "https": proxy_url},
                timeout=10
            )
            if response.status_code == 200:
                logger.info(f"Proxy test successful: {proxy.host}:{proxy.port}")
                return True
            else:
                logger.warning(f"Proxy test failed with status {response.status_code}: {proxy.host}:{proxy.port}")
                return False
        except Exception as e:
            logger.error(f"Proxy test error for {proxy.host}:{proxy.port}: {e}")
            return False

    def get_working_proxy(self) -> Optional[ProxyInfo]:
        """Get the next working proxy, testing each one."""
        max_attempts = len(self.proxies) * 2  # Try twice through the list
        attempts = 0
        
        while attempts < max_attempts:
            proxy = self.get_next_proxy()
            if proxy is None:
                break
                
            if self.test_proxy(proxy):
                return proxy
            else:
                self.mark_proxy_failed(proxy)
                
            attempts += 1
        
        logger.error("No working proxies found")
        return None

    def set_location(self, country: str = None, city: str = None):
        """Change the geographic location for proxies."""
        self.country = country
        self.city = city
        logger.info(f"Updated proxy location: country={country}, city={city}")
        # Force refresh on next get_next_proxy call
        self.last_refresh = 0