import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import random
from collections import deque
import json
import os
import hashlib
from datetime import datetime

class WebsiteScraper:
    CONFIG_FILE = 'scraper_config.json'
    
    def __init__(self):
        """Initialize the scraper with interactive prompts and saved settings"""
        self.load_config()
        self.session = requests.Session()
        self.visited = set()
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set headers to mimic a browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',  # Do Not Track
            'Sec-GPC': '1'  # Global Privacy Control
        }
        
        # Privacy-focused session settings
        self.session.trust_env = False  # Ignore system proxy settings
        self.session.cookies.clear()  # No cookies
        
        if self.use_vpn:
            self._check_vpn_connection()
            
        if self.proxy:
            self._setup_proxy()
    
    def _get_public_ip(self):
        """Get current public IP through the current session"""
        try:
            return self.session.get('https://api.ipify.org', timeout=5).text
        except Exception as e:
            print(f"Could not determine public IP: {e}")
            return "UNKNOWN (network error)"
    
    def _check_vpn_connection(self):
        """Enhanced VPN check with clear warnings"""
        try:
            print("\n" + "="*50)
            print("VPN STATUS CHECK".center(50))
            print("="*50)
            
            # Get current public IP without VPN
            with requests.Session() as temp_session:
                original_ip = temp_session.get('https://api.ipify.org', timeout=5).text
            
            # Get current public IP through session (with VPN if configured)
            current_ip = self._get_public_ip()
            
            if original_ip == current_ip:
                print("\n WARNING: VPN NOT ACTIVE".center(50))
                print(f"\nYour REAL IP address is exposed: {current_ip}")
                print("\nRecommendations:")
                print("- Connect to your VPN before continuing")
                print("- Consider using the proxy option")
                print("- Press Ctrl+C now to abort if privacy is critical")
                
                # Countdown to continue
                print("\nContinuing in 5 seconds (press Ctrl+C to cancel)...")
                for i in range(5, 0, -1):
                    print(f"{i}...", end=' ', flush=True)
                    time.sleep(1)
                print("\n")
            else:
                print(f"\nVPN appears active. Original IP: {original_ip}, Current IP: {current_ip}")
                
        except Exception as e:
            print(f"\nCould not verify VPN status: {e}")
            print("Proceeding with potentially UNPROTECTED connection")
        
        print("="*50 + "\n")
    
    def load_config(self):
        """Load or create configuration with defaults"""
        defaults = {
            'base_url': 'https://example.com',
            'max_depth': 1,
            'use_vpn': True,
            'proxy': None,
            'delay': 2,
            'cache_dir': 'scraper_cache',
            'cache_expiry_hours': 24
        }
        
        try:
            with open(self.CONFIG_FILE, 'r') as f:
                config = json.load(f)
            # Merge with defaults in case new settings were added
            self.config = {**defaults, **config}
        except FileNotFoundError:
            self.config = defaults
            self.save_config()
        
        # Set attributes from config
        for key, value in self.config.items():
            setattr(self, key, value)
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def prompt_settings(self):
        """Interactive prompt for scraper settings"""
        print("\n=== Website Scraper Configuration ===")
        
        # URL
        self.base_url = input(f"Enter starting URL [{self.base_url}]: ") or self.base_url
        
        # Depth
        while True:
            depth_input = input(f"Enter maximum depth (0-10) [{self.max_depth}]: ") or str(self.max_depth)
            if depth_input.isdigit() and 0 <= int(depth_input) <= 10:
                self.max_depth = int(depth_input)
                break
            print("Please enter a number between 0 and 10")
        
        # VPN
        vpn_input = input(f"Use VPN? (y/n) [{'y' if self.use_vpn else 'n'}]: ").lower()
        self.use_vpn = vpn_input == 'y' if vpn_input in ('y', 'n') else self.use_vpn
        
        # Proxy
        proxy_input = input(f"Enter proxy (leave blank to keep current) [{self.proxy or 'none'}]: ")
        if proxy_input:
            self.proxy = proxy_input if proxy_input.lower() != 'none' else None
        
        # Delay
        while True:
            delay_input = input(f"Enter delay between requests (seconds) [{self.delay}]: ") or str(self.delay)
            try:
                self.delay = max(0, float(delay_input))
                break
            except ValueError:
                print("Please enter a valid number")
        
        # Cache settings
        self.cache_dir = input(f"Enter cache directory [{self.cache_dir}]: ") or self.cache_dir
        while True:
            expiry_input = input(f"Enter cache expiry (hours) [{self.cache_expiry_hours}]: ") or str(self.cache_expiry_hours)
            try:
                self.cache_expiry_hours = max(1, int(expiry_input))
                break
            except ValueError:
                print("Please enter a valid integer")
        
        # Update config for next run
        self.config.update({
            'base_url': self.base_url,
            'max_depth': self.max_depth,
            'use_vpn': self.use_vpn,
            'proxy': self.proxy,
            'delay': self.delay,
            'cache_dir': self.cache_dir,
            'cache_expiry_hours': self.cache_expiry_hours
        })
        self.save_config()
    
    def _setup_proxy(self):
        """Configure proxy for requests"""
        if self.proxy:
            self.session.proxies = {
                'http': self.proxy,
                'https': self.proxy,
            }
            print(f"Proxy configured: {self.proxy.split('@')[-1]}")
    
    def _is_valid_url(self, url):
        """Check if URL is valid and belongs to the same domain"""
        parsed = urlparse(url)
        base_parsed = urlparse(self.base_url)
        return parsed.netloc == base_parsed.netloc and parsed.scheme in ('http', 'https')
    
    def _get_cache_filename(self, url):
        """Generate a consistent cache filename for a URL"""
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"{url_hash}.json")
    
    def _load_from_cache(self, url):
        """Load page content from cache if available and not expired"""
        cache_file = self._get_cache_filename(url)
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                
            # Check if cache is expired
            cache_time = datetime.strptime(cache_data['timestamp'], '%Y-%m-%d %H:%M:%S')
            age_seconds = (datetime.now() - cache_time).total_seconds()
            
            if age_seconds > self.cache_expiry_hours * 3600:
                print(f"Cache expired for {url}")
                return None
                
            return cache_data['content']
        except Exception as e:
            print(f"Error reading cache for {url}: {e}")
            return None
    
    def _save_to_cache(self, url, content):
        """Save page content to cache"""
        cache_file = self._get_cache_filename(url)
        
        cache_data = {
            'url': url,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'content': content
        }
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            print(f"Error saving cache for {url}: {e}")
    
    def _get_page_content(self, url):
        """Get page content either from cache or by making a request"""
        # Try to load from cache first
        cached_content = self._load_from_cache(url)
        if cached_content is not None:
            print(f"Loaded from cache: {url}")
            return cached_content
        
        # If not in cache or expired, make a request
        try:
            time.sleep(self.delay + random.uniform(0, 1))  # Random delay
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Save to cache before returning
            self._save_to_cache(url, response.text)
            return response.text
            
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def _get_links(self, url):
        """Extract all links from a page"""
        page_content = self._get_page_content(url)
        if not page_content:
            return set()
        
        soup = BeautifulSoup(page_content, 'html.parser')
        links = set()
        
        for link in soup.find_all('a', href=True):
            absolute_url = urljoin(url, link['href'])
            if self._is_valid_url(absolute_url):
                links.add(absolute_url)
        
        return links
    
    def scrape(self):
        """Start scraping with controlled depth"""
        queue = deque([(self.base_url, 0)])
        results = []
        
        while queue:
            current_url, current_depth = queue.popleft()
            
            if current_url in self.visited or current_depth > self.max_depth:
                continue
                
            self.visited.add(current_url)
            print(f"Scraping: {current_url} (Depth: {current_depth})")
            
            # Get page content and links
            links = self._get_links(current_url)
            results.append({
                'url': current_url,
                'depth': current_depth,
                'links': list(links)
            })
            
            # Add new links to queue with increased depth
            if current_depth < self.max_depth:
                for link in links:
                    if link not in self.visited:
                        queue.append((link, current_depth + 1))
        
        return results

    def clear_cache(self):
        """Clear all cached pages"""
        try:
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            print("Cache cleared successfully.")
        except Exception as e:
            print(f"Error clearing cache: {e}")

def main():
    """Main interactive function"""
    scraper = WebsiteScraper()
    
    print("=== Website Scraper ===")
    print(f"Current settings (saved in {scraper.CONFIG_FILE}):")
    print(f" - URL: {scraper.base_url}")
    print(f" - Max Depth: {scraper.max_depth}")
    print(f" - VPN: {'Enabled' if scraper.use_vpn else 'Disabled'}")
    print(f" - Proxy: {scraper.proxy or 'None'}")
    print(f" - Delay: {scraper.delay} seconds")
    print(f" - Cache: {scraper.cache_dir} (expires after {scraper.cache_expiry_hours} hours)")
    
    change_settings = input("\nChange settings before scraping? (y/n) [n]: ").lower() == 'y'
    if change_settings:
        scraper.prompt_settings()
    
    # Offer to clear cache
    if input("\nClear cache before scraping? (y/n) [n]: ").lower() == 'y':
        scraper.clear_cache()
    
    print("\nStarting scraping...")
    scraped_data = scraper.scrape()
    
    # Print results
    print("\nScraping Results:")
    for page in scraped_data:
        print(f"\nURL: {page['url']} (Depth: {page['depth']})")
        print(f"Found {len(page['links'])} links:")
        for link in page['links']:
            print(f"  - {link}")
    
    print(f"\nTotal pages scraped: {len(scraped_data)}")

if __name__ == "__main__":
    main()