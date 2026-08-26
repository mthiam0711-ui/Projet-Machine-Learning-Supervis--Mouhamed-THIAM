"""Télécharge le dataset Telco Customer Churn dans data/.

Usage : python data/download_data.py
"""
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
OUT_PATH = Path(__file__).parent / "Telco-Customer-Churn.csv"


def main():
    print(f"Téléchargement depuis {URL} ...")
    urllib.request.urlretrieve(URL, OUT_PATH)
    print(f"Dataset enregistré dans {OUT_PATH}")


if __name__ == "__main__":
    main()
