import json
import os

import requests


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_URL = os.environ.get(
    "GEMINI_API_URL",
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
GEMINI_TIMEOUT = int(os.environ.get("GEMINI_TIMEOUT", 20))


EMERGENCY_KEYWORDS = [
    "bunuh diri",
    "mengakhiri hidup",
    "menyakiti diri",
    "menyakiti bayi",
    "tidak ingin hidup",
    "ingin mati",
    "self harm",
    "suicide",
]

IN_SCOPE_KEYWORDS = [
    "ibu",
    "istri",
    "ayah",
    "bapak",
    "pasangan",
    "bayi",
    "anak",
    "postpartum",
    "pasca melahirkan",
    "melahirkan",
    "kehamilan",
    "hamil",
    "menyusui",
    "asi",
    "depresi",
    "cemas",
    "sedih",
    "menangis",
    "emosi",
    "marah",
    "tidur",
    "lelah",
    "kewalahan",
    "stres",
    "skrining",
    "rekomendasi",
    "dukungan",
    "mood",
    "mental",
    "kesehatan",
    "bidan",
    "psikolog",
    "dokter",
    "puskesmas",
]


def has_emergency_signal(message):
    lowered = message.lower()
    return any(keyword in lowered for keyword in EMERGENCY_KEYWORDS)


def is_in_scope(message):
    lowered = message.lower()
    return any(keyword in lowered for keyword in IN_SCOPE_KEYWORDS)


def build_out_of_scope_reply(user_role):
    subject = "mendukung Ibu" if user_role == "father" else "mendampingi Ibu"
    return {
        "source": "local",
        "priority": "normal",
        "reply": (
            f"Saya paham pertanyaannya, tetapi saya punya batasan dan hanya bisa membantu dalam konteks Nurtura: "
            f"kesehatan mental postpartum, hasil skrining, perawatan emosi setelah melahirkan, dan cara {subject}. "
            "Kalau Ayah ingin, ceritakan kondisi Ibu atau situasi di rumah yang sedang membuat Ayah khawatir."
        ),
        "suggested_actions": [
            "Tanyakan tentang hasil skrining atau rekomendasi Ibu.",
            "Ceritakan perubahan emosi, tidur, makan, atau kecemasan yang terlihat.",
            "Gunakan pertanyaan seputar cara memberi dukungan kepada Ibu."
        ],
        "disclaimer": "Chatbot ini bersifat pendamping awal, bukan diagnosis medis."
    }


def build_local_reply(message, user_role, context):
    emergency = has_emergency_signal(message)
    latest_prediction = context.get("latest_prediction") or {}
    latest_result = latest_prediction.get("result")

    if not emergency and not is_in_scope(message):
        return build_out_of_scope_reply(user_role)

    if emergency:
        if user_role == "father":
            reply = (
                "Jika ada risiko ibu menyakiti diri sendiri, bayi, atau merasa tidak aman, jangan biarkan ibu sendirian. "
                "Temani dengan tenang, hubungi keluarga terdekat, dan segera cari bantuan darurat atau fasilitas kesehatan terdekat."
            )
        else:
            reply = (
                "Saya ikut khawatir mendengarnya. Jika Ibu merasa ingin menyakiti diri, bayi, atau merasa tidak aman, "
                "segera minta seseorang menemani sekarang dan hubungi layanan darurat atau fasilitas kesehatan terdekat."
            )

        return {
            "source": "local",
            "priority": "darurat",
            "reply": reply,
            "suggested_actions": [
                "Jangan sendirian saat ini.",
                "Hubungi pasangan, keluarga, atau orang terdekat.",
                "Cari bantuan darurat atau fasilitas kesehatan terdekat."
            ],
            "disclaimer": "Chatbot ini bersifat pendamping awal, bukan diagnosis medis."
        }

    if user_role == "father":
        reply = (
            "Saya bisa bantu Bapak memahami kondisi ibu dan mencari cara mendukungnya. "
            "Dengarkan tanpa menghakimi, tawarkan bantuan konkret seperti menjaga bayi atau pekerjaan rumah, "
            "dan ajak ibu mencari bantuan profesional bila keluhan terasa berat atau menetap."
        )
    else:
        reply = (
            "Saya bisa menemani Ibu memahami perasaan ini pelan-pelan. Coba mulai dari satu langkah kecil: "
            "ceritakan kondisi hari ini ke orang yang dipercaya, ambil waktu istirahat singkat, dan catat perubahan suasana hati."
        )

    if latest_result:
        reply += f" Hasil skrining terakhir tercatat: {latest_result}."

    return {
        "source": "local",
        "priority": "perhatian" if latest_result == "Beresiko Depresi" else "normal",
        "reply": reply,
        "suggested_actions": [
            "Ceritakan kondisi kepada orang yang dipercaya.",
            "Catat perubahan suasana hati, tidur, dan makan.",
            "Pertimbangkan konsultasi dengan tenaga kesehatan bila keluhan mengganggu aktivitas."
        ],
        "disclaimer": "Chatbot ini bersifat pendamping awal, bukan diagnosis medis."
    }


def extract_gemini_text(response_json):
    candidates = response_json.get("candidates", [])
    if not candidates:
        return None

    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [part.get("text", "") for part in parts if part.get("text")]

    return "\n".join(texts).strip() if texts else None


def parse_chatbot_json(text):
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


def build_chatbot_prompt(message, user_role, context, history):
    role_instruction = (
        "User adalah ayah/pasangan. Jawab dengan fokus membantu ayah mendukung ibu secara praktis, hangat, dan tidak menyalahkan."
        if user_role == "father"
        else "User adalah ibu postpartum. Jawab langsung kepada ibu dengan empatik, menenangkan, dan praktis."
    )

    payload = {
        "user_role": user_role,
        "message": message,
        "context": context,
        "recent_history": history[-8:] if isinstance(history, list) else []
    }

    return (
        "Anda adalah chatbot Nurtura untuk pendamping kesehatan mental postpartum. "
        f"{role_instruction} "
        "Gaya bicara harus terasa manusiawi: hangat, natural, tidak kaku, tidak seperti template, dan tidak terlalu panjang. "
        "Validasi dulu perasaan user dalam satu kalimat singkat, lalu berikan langkah praktis. "
        "Jangan mengulang disclaimer di dalam reply karena field disclaimer sudah tersedia. "
        "Batas konteks Anda hanya kesehatan mental postpartum, hasil skrining Nurtura, perawatan emosi setelah melahirkan, "
        "dukungan ayah/pasangan, relasi keluarga dekat, bayi, dan rujukan ke tenaga kesehatan. "
        "Jika user bertanya di luar konteks tersebut, jawab dengan sopan bahwa Anda memiliki batasan di luar konteks Nurtura, "
        "lalu arahkan user kembali ke topik skrining, kondisi ibu, bayi, atau dukungan pasangan. "
        "Jangan membuat diagnosis medis, jangan memberi instruksi obat, dan jangan menggantikan tenaga kesehatan. "
        "Jika pesan mengandung niat menyakiti diri, menyakiti bayi, bunuh diri, atau kondisi tidak aman, beri prioritas darurat: "
        "minta user segera ditemani dan mencari bantuan darurat/fasilitas kesehatan terdekat. "
        "Gunakan konteks hasil skrining bila tersedia, tetapi jangan menghakimi. "
        "Untuk pertanyaan normal, isi suggested_actions maksimal 3 item dan buat itemnya konkret. "
        "Untuk pertanyaan di luar konteks, isi suggested_actions dengan 2-3 cara bertanya yang masih sesuai konteks Nurtura. "
        "Balas hanya JSON valid tanpa markdown dengan struktur: "
        "{"
        "\"source\":\"gemini\","
        "\"priority\":\"normal|perhatian|darurat\","
        "\"reply\":\"...\","
        "\"suggested_actions\":[\"...\"],"
        "\"disclaimer\":\"Chatbot ini bersifat pendamping awal, bukan diagnosis medis.\""
        "}. "
        f"Data percakapan: {json.dumps(payload, ensure_ascii=False)}"
    )

print("GEMINI_API_KEY =", GEMINI_API_KEY)
def get_chatbot_reply(message, user_role, context=None, history=None):
    context = context or {}
    history = history or []
    fallback = build_local_reply(message, user_role, context)

    if not GEMINI_API_KEY:
        fallback["ai_error"] = "GEMINI_API_KEY belum disetel; menggunakan jawaban lokal."
        return fallback

    try:
        url = GEMINI_API_URL.format(model=GEMINI_MODEL)
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": build_chatbot_prompt(message, user_role, context, history)}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.5,
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

        chatbot_reply = parse_chatbot_json(extract_gemini_text(response.json()))
        if not chatbot_reply:
            raise ValueError("Respons Gemini tidak berisi JSON chatbot yang valid")

        chatbot_reply["source"] = "gemini"
        chatbot_reply.setdefault("priority", "normal")
        chatbot_reply.setdefault("suggested_actions", [])
        chatbot_reply.setdefault("disclaimer", "Chatbot ini bersifat pendamping awal, bukan diagnosis medis.")

        return chatbot_reply
    except Exception as e:
        fallback["ai_error"] = str(e)
        return fallback
