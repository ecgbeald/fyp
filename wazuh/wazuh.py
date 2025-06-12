# wrapper code to call the Wazuh logtest API to classify logs

import subprocess
import json

def authenticate(url, key):
    auth_cmd = [
        "curl", "-u", key,
        "-k", "-X", "POST",
        f"https://{url}:55000/security/user/authenticate?raw=true"
    ]
    token = subprocess.check_output(auth_cmd).decode().strip()
    return token

def test_success_connection(url, token):
    get_cmd = [
        "curl", "-k", "-X", "GET",
        f"https://{url}:55000/?pretty=true",
        "-H", f"Authorization: Bearer {token}"
    ]
    output = subprocess.check_output(get_cmd).decode()
    if "Wazuh" in output:
        print("Connection successful!")
    else:
        print("Failed to connect to Wazuh server.")

def build_json_request(log, token):
    return {
        "token": token,
        "event": log,
        "log_format": "apache",
        "location": "stdin",
    }
    
def analyze_log_with_wazuh(log_data, url, token):
    data_str = json.dumps(log_data)
    response = subprocess.run([
        "curl", "-k", "-X", "PUT",
        f"https://{url}:55000/logtest?pretty=true",
        "-H", f"Authorization: Bearer {token}",
        "-H", "Content-Type: application/json",
        "-d", data_str
    ], capture_output=True, text=True)

    json_data = json.loads(response.stdout)
    return json_data

def interpret_wazuh_response(json_data):
    try:
        rule = json_data['data']['output']['rule']
        groups = rule.get('groups', [])
        is_attack = "attack" in groups

        # print("Rule ID:", rule.get("id"))
        # print("Description:", rule.get("description"))
        # print("Groups:", groups)

        if is_attack:
            print("attack")
        else:
            print("not attack.")
        
        return is_attack, rule.get("description")
    except Exception as e:
        print("Failed to parse Wazuh response:", e)
        return False, None
