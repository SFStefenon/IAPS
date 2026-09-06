import requests
from pathlib import Path

output_dir = Path("dados_hidrologicos")
output_dir.mkdir(exist_ok=True)

# Download monthly ONS data
for year in range(2025, 2027):
    for month in range(1, 13):
        filename = f"DADOS_HIDROLOGICOS_HO_{year}_{month:02d}.csv"
        url = (
            "https://ons-aws-prod-opendata.s3.amazonaws.com/"
            f"dataset/dados_hidrologicos_ho/{filename}"
        )

        response = requests.get(url, timeout=60)

        if response.ok:
            (output_dir / filename).write_bytes(response.content)
            print(f"Downloaded: {filename}")
        else:
            print(f"Not available: {filename} ({response.status_code})")

# Download the Tucuruí dataset from GitHub
url = (
    "https://raw.githubusercontent.com/SFStefenon/"
    "NaturalFlowforHydroelectricity/main/tucurui.csv"
)

response = requests.get(url, timeout=60)
response.raise_for_status()

Path("tucurui.csv").write_bytes(response.content)
print("Downloaded: tucurui.csv")
