import subprocess
import json

# curl to the targeting server, for testing "malicious" endpoints
# the endpoints will *not* be included in the repo for obvious reasons, but I scraped them from public github repositories
server_url = "[MASK]"

def test_endpoints(endpoints):
    for endpoint in endpoints:
        # Form the complete URL
        url = f"{server_url}{endpoint}"
        
        # Perform the curl request and capture the response
        try:
            # Run curl command, HEAD
            response = subprocess.run(['curl', '-I', url], capture_output=True, text=True)
            
            # Print the URL and the response status code
            print(f"Testing: {url}")
            print(f"Response Code: {response.returncode}")
            if response.returncode == 0:
                print(f"Headers:\n{response.stdout}")
            else:
                print(f"Error: {response.stderr}")
            print("-" * 50)
        except Exception as e:
            print(f"Error occurred while testing {url}: {str(e)}")

with open("requests.json", "r") as file:
    data = json.load(file)

# Extract endpoints into a variable
endpoints = data.get("endpoints", [])

# Print the endpoints to verify
print(endpoints)

# Submit to server_url
test_endpoints(endpoints)