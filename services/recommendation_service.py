import json
import os

import requests

from utils.screening_features import HIGH_RISK_FIELDS, summarize_answers


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_URL = os.environ.get(
    "GEMINI_API_URL",
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
GEMINI_TIMEOUT = int(os.environ.get("GEMINI_TIMEOUT", 20))


def extract_focus_areas(answer_summary, limit=4):
    priority_items = sorted(
        answer_summary,
        key=lambda item: (
            item["field"] in HIGH_RISK_FIELDS,
            item["score"]
        ),
        reverse=True
    )

    return [
        {
            "field": item["field"],
            "label": item["label"],
            "answer": item["answer"],
            "score": item["score"]
        }
        for item in priority_items
        if item["score"] > 0
    ][:limit]


def has_suicide_signal(answers, answer_summary):
    suicide_answer = str(answers.get("percobaan_bunuh_diri", "")).strip().lower()
    suicide_score = next(
        (item["score"] for item in answer_summary if item["field"] == "percobaan_bunuh_diri"),
        0
    )

    return suicide_score > 0 or suicide_answer not in ["", "no", "not at all"]


def build_local_recommendation(result, answers, features, cluster):
    answer_summary = summarize_answers(answers, features)
    focus_areas = extract_focus_areas(answer_summary)
    total_score = int(sum(features))
    suicide_signal = has_suicide_signal(answers, answer_summary)
    is_risky = result == "Beresiko Depresi"

    if suicide_signal:
        priority = "darurat"
        summary = (
            "Jawaban menunjukkan adanya sinyal keselamatan diri. Ibu perlu ditemani dan "
            "segera diarahkan ke bantuan profesional atau layanan darurat terdekat."
        )
    elif is_risky:
        priority = "tinggi"
        summary = (
            "Hasil skrining menunjukkan risiko depresi. Rekomendasi utama adalah mencari "
            "dukungan keluarga dan menjadwalkan konsultasi dengan tenaga kesehatan."
        )
    elif total_score >= 10:
        priority = "sedang"
        summary = (
            "Beberapa keluhan masih cukup terasa. Ibu disarankan memantau kondisi, "
            "mengatur waktu istirahat, dan melibatkan pasangan atau keluarga."
        )
    else:
        priority = "rendah"
        summary = (
            "Hasil skrining tidak menunjukkan risiko depresi saat ini. Tetap jaga rutinitas "
            "pemulihan, istirahat, dan komunikasi dengan orang terdekat."
        )

    action_steps = [
        "Ceritakan kondisi hari ini kepada pasangan atau keluarga yang dipercaya.",
        "Atur satu waktu istirahat singkat dan kurangi beban pekerjaan rumah yang tidak mendesak.",
        "Catat perubahan suasana hati, tidur, makan, dan kecemasan selama beberapa hari ke depan."
    ]

    if is_risky or priority in ["sedang", "tinggi", "darurat"]:
        action_steps.append("Pertimbangkan konsultasi dengan bidan, psikolog, dokter, atau fasilitas kesehatan terdekat.")

    if suicide_signal:
        action_steps.insert(0, "Jangan biarkan ibu sendirian; minta pendampingan segera dari orang terdekat.")

    return {
        "source": "local",
        "priority": priority,
        "summary": summary,
        "focus_areas": focus_areas,
        "action_steps": action_steps,
        "partner_support": [
            "Dengarkan tanpa menghakimi dan tanyakan bantuan konkret yang dibutuhkan.",
            "Bantu pengasuhan bayi, pekerjaan rumah, dan jadwal istirahat ibu.",
            "Temani ibu bila perlu menghubungi tenaga kesehatan."
        ],
        "professional_help": (
            "Hubungi bidan, psikolog, dokter, puskesmas, rumah sakit, atau layanan kesehatan terdekat "
            "bila keluhan menetap, memburuk, atau mengganggu aktivitas harian."
        ),
        "emergency_note": (
            "Jika muncul pikiran menyakiti diri, menyakiti bayi, atau merasa tidak aman, segera cari bantuan "
            "darurat dan jangan menunggu jadwal konsultasi."
        ),
        "disclaimer": "Rekomendasi ini bersifat pendamping skrining, bukan diagnosis medis."
    }


def extract_gemini_text(response_json):
    candidates = response_json.get("candidates", [])
    if not candidates:
        return None

    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [part.get("text", "") for part in parts if part.get("text")]

    return "\n".join(texts).strip() if texts else None


def parse_recommendation_json(text):
    if not text:
        return None

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    return json.loads(cleaned[start:end + 1])


def build_gemini_prompt(result, cluster, answers, features):
    answer_summary = summarize_answers(answers, features)

    payload = {
        "screening_result": result,
        "cluster": int(cluster),
        "total_score": int(sum(features)),
        "answers": answer_summary
    }

    return (
        "Anda adalah asisten rekomendasi kesehatan mental postpartum untuk aplikasi Nurtura. "
        "Buat rekomendasi personal dalam Bahasa Indonesia berdasarkan hasil skrining ibu. "
        "Gunakan nada hangat, praktis, tidak menghakimi, dan jangan membuat diagnosis medis. "
        "Jika ada sinyal percobaan bunuh diri atau keselamatan diri, beri prioritas darurat dan arahkan untuk segera didampingi serta mencari bantuan darurat/profesional. "
        "Balas hanya JSON valid tanpa markdown dengan struktur: "
        "{"
        "\"source\":\"gemini\","
        "\"priority\":\"rendah|sedang|tinggi|darurat\","
        "\"summary\":\"...\","
        "\"focus_areas\":[{\"field\":\"...\",\"label\":\"...\",\"answer\":\"...\",\"score\":0}],"
        "\"action_steps\":[\"...\"],"
        "\"partner_support\":[\"...\"],"
        "\"professional_help\":\"...\","
        "\"emergency_note\":\"...\","
        "\"disclaimer\":\"Rekomendasi ini bersifat pendamping skrining, bukan diagnosis medis.\""
        "}. "
        f"Data skrining: {json.dumps(payload, ensure_ascii=False)}"
    )


def get_ai_recommendation(result, answers, features, cluster):
    fallback = build_local_recommendation(result, answers, features, cluster)

    if not GEMINI_API_KEY:
        fallback["ai_error"] = "GEMINI_API_KEY belum disetel; menggunakan rekomendasi lokal."
        return fallback

    try:
        url = GEMINI_API_URL.format(model=GEMINI_MODEL)
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": build_gemini_prompt(result, cluster, answers, features)}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "responseMimeType": "application/json"
            }
        }

        session = requests.Session()
        session.trust_env = False

        response = session.post(
            url,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": GEMINI_API_KEY
            },
            json=payload,
            timeout=GEMINI_TIMEOUT
        )
        response.raise_for_status()

        recommendation = parse_recommendation_json(extract_gemini_text(response.json()))
        if not recommendation:
            raise ValueError("Respons Gemini tidak berisi JSON rekomendasi yang valid")

        recommendation["source"] = "gemini"
        recommendation.setdefault("disclaimer", "Rekomendasi ini bersifat pendamping skrining, bukan diagnosis medis.")
        return recommendation
    except Exception as e:
        fallback["ai_error"] = str(e)
        return fallback
