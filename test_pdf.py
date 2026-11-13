import requests

url = "http://127.0.0.1:8000/upload-pds"

params = {
    "position_title": "Information Technology Officer II",
    "education": 4,
    "experience": 5,
    "training_hours": 40,
    "eligibility_level": "professional",
}

#  Change this to wherever your PDF actually is
file_path = r"C:\capstone\pds-extractor-api\CSC (Master)-(4yrs)-(PROF)-(30TH).pdf"

with open(file_path, "rb") as f:
    r = requests.post(url, files={"file": f}, params=params)

print("Status:", r.status_code)
print(r.json())