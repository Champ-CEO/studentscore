# download_db.py
import requests
import os

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# URL of the database
url = "https://techassessment.blob.core.windows.net/aiap-preparatory-bootcamp/score.db"

# Download the database
response = requests.get(url)
with open("data/raw/score.db", "wb") as f:
    f.write(response.content)

print("Database downloaded successfully!")
