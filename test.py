import requests
import json

url = "http://localhost:8080/predict"

# Test 1: Invalid mother_id
print("=" * 50)
print("Test 1: Invalid mother_id (tidak ada di database)")
print("=" * 50)
data1 = {
    "answers": {
        "perasaan_sedih_atau_mudah_menangis": "Yes",
        "mudah_marah_terhadap_bayi_dan_pasangan": "Not at all",
        "kesulitan_tidur_di_malam_hari": "Often",
        "kesulitan_konsentrasi_atau_mengambil_keputusan": "Sometimes",
        "makan_berlebihan_atau_kehilangan_nafsu_makan": "Not at all",
        "merasa_cemas": "Yes",
        "perasaan_bersalah": "Maybe",
        "kesulitan_membangun_ikatan_dengan_bayi": "Not at all",
        "percobaan_bunuh_diri": "No"
    },
    "mother_id": "000000000000000000000000"  # Invalid ID
}

try:
    response = requests.post(url, json=data1)
    print("Status Code:", response.status_code)
    print("Response:", json.dumps(response.json(), indent=2))
except Exception as e:
    print("Error:", str(e))

# Test 2: Valid mother_id (assuming this exists)
print("\n" + "=" * 50)
print("Test 2: Valid mother_id")
print("=" * 50)
data2 = {
    "answers": {
        "perasaan_sedih_atau_mudah_menangis": "Yes",
        "mudah_marah_terhadap_bayi_dan_pasangan": "Not at all",
        "kesulitan_tidur_di_malam_hari": "Often",
        "kesulitan_konsentrasi_atau_mengambil_keputusan": "Sometimes",
        "makan_berlebihan_atau_kehilangan_nafsu_makan": "Not at all",
        "merasa_cemas": "Yes",
        "perasaan_bersalah": "Maybe",
        "kesulitan_membangun_ikatan_dengan_bayi": "Not at all",
        "percobaan_bunuh_diri": "No"
    },
    "mother_id": "507f1f77bcf86cd799439011"
}

try:
    response = requests.post(url, json=data2)
    print("Status Code:", response.status_code)
    print("Response:", json.dumps(response.json(), indent=2))
except Exception as e:
    print("Error:", str(e))