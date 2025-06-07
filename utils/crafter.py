# Craft malicious requests to test Apache Server

import subprocess
import json
import argparse

# curl to the targeting server, for testing "malicious" endpoints
# the endpoints will *not* be included in the repo for obvious reasons, but I scraped them from public github repositories
server_url = "[MASK]"


def test_endpoints(endpoints):
    for endpoint in endpoints:
        # Form the complete URL
        url = f"{server_url}{endpoint}"

        try:
            response = subprocess.run(
                ["curl", "-I", url], capture_output=True, text=True
            )

            print(f"Testing: {url}")
            print(f"Response Code: {response.returncode}")
            if response.returncode == 0:
                print(f"Headers:\n{response.stdout}")
            else:
                print(f"Error: {response.stderr}")
            print("-" * 50)
        except Exception as e:
            print(f"Error occurred while testing {url}: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a JSON file.")
    parser.add_argument("json_file", type=str, help="Path to the JSON file")
    args = parser.parse_args()
    json_file = args.json_file

    print(f"Using JSON file: {json_file}")
    
    try:
        with open(json_file, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"{json_file} file not found.")
        exit(1)