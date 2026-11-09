import requests

url = "https://object-arbutus.alliancecan.ca/swift/v1/14c95234f6cd4a21a47deafe20cce2a7/tgb/tgbl-flight-v2.zip"

print("Testing connection to TGB server...")
try:
    r = requests.head(url, timeout=30)
    print(f"Status: {r.status_code}")
    cl = r.headers.get("Content-Length", "unknown")
    print(f"Content-Length: {cl} bytes ({int(cl)/1e9:.2f} GB)" if cl != "unknown" else f"Content-Length: {cl}")
except requests.exceptions.ConnectTimeout:
    print("TIMEOUT - Server unreachable from this network")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
