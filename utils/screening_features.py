MAPPING = {
    "Not at all": 0,
    "No": 0,
    "Maybe": 1,
    "Not interested to say": 1,
    "Sometimes": 2,
    "Often": 3,
    "Yes": 4,
    "Two or more days a week": 4
}

FIELD_TO_INDEX = {
    "perasaan_sedih_atau_mudah_menangis": 0,
    "mudah_marah_terhadap_bayi_dan_pasangan": 1,
    "kesulitan_tidur_di_malam_hari": 2,
    "kesulitan_konsentrasi_atau_mengambil_keputusan": 3,
    "makan_berlebihan_atau_kehilangan_nafsu_makan": 4,
    "merasa_cemas": 5,
    "perasaan_bersalah": 6,
    "kesulitan_membangun_ikatan_dengan_bayi": 7,
    "percobaan_bunuh_diri": 8
}

FEATURE_LABELS = {
    "perasaan_sedih_atau_mudah_menangis": "Perasaan sedih atau mudah menangis",
    "mudah_marah_terhadap_bayi_dan_pasangan": "Mudah marah terhadap bayi dan pasangan",
    "kesulitan_tidur_di_malam_hari": "Kesulitan tidur di malam hari",
    "kesulitan_konsentrasi_atau_mengambil_keputusan": "Kesulitan konsentrasi atau mengambil keputusan",
    "makan_berlebihan_atau_kehilangan_nafsu_makan": "Makan berlebihan atau kehilangan nafsu makan",
    "merasa_cemas": "Merasa cemas",
    "perasaan_bersalah": "Perasaan bersalah",
    "kesulitan_membangun_ikatan_dengan_bayi": "Kesulitan membangun ikatan dengan bayi",
    "percobaan_bunuh_diri": "Percobaan bunuh diri"
}

HIGH_RISK_FIELDS = {
    "perasaan_sedih_atau_mudah_menangis",
    "merasa_cemas",
    "perasaan_bersalah",
    "kesulitan_membangun_ikatan_dengan_bayi",
    "percobaan_bunuh_diri"
}


def build_features(answers):
    features = [0] * len(FIELD_TO_INDEX)

    for field, answer in answers.items():
        if field in FIELD_TO_INDEX:
            features[FIELD_TO_INDEX[field]] = MAPPING.get(answer, 0)

    return features


def summarize_answers(answers, features):
    items = []

    for field, index in FIELD_TO_INDEX.items():
        score = int(features[index])
        items.append({
            "field": field,
            "label": FEATURE_LABELS[field],
            "answer": str(answers.get(field, "")),
            "score": score
        })

    return items
