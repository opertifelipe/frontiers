import requests

url = "https://localhost:8082"

payload = "client_id=CLIENT_ID&client_secret=CLIENT_SECRET&grant_type=client_credentials&scope=emsi_open"
headers = {'Content-Type': 'application/x-www-form-urlencoded'}

response = requests.request("POST", url, data=payload, headers=headers)

print(response.text)