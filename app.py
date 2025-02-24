import requests

url = "https://23e5-34-46-195-63.ngrok-free.app/predict"
# payload = {
#     "static_features": [[2021, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
#     "sequence_features": [[[8.50,20],[8.73, 22], [8.45, 22], [9.09, 22], [9.22, 23],[9.13, 31]]],
#     "extra_credit": [20.0]
# }
payload = {
    "static_features": [[2021, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
    "sequence_features": [[[8.50,20],[8.73, 22],[8.45, 22], [9.09, 22], [9.22, 23],[9.13, 31]]],
    "extra_credit": [20.0]
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
