import os
import time
import json
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
import pyttsx3
import requests
from groq import Groq
from bs4 import BeautifulSoup   

# =========================
# CONFIG
# =========================
SAMPLE_RATE = 16000
AUDIO_FILE = "temp_audio.wav"

client = Groq(api_key=os.getenv("groqapi"))
WEATHER_API_KEY = os.getenv("weatherapi")

# =========================
# TEXT TO SPEECH
# =========================
engine = pyttsx3.init()
engine.setProperty('rate', 170)
engine.setProperty('volume', 1.0)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# =========================
# SILENCE-BASED RECORDING
# =========================
def record_audio():
    print("Waiting for speech...")

    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        callback=callback
    )

    silence_threshold = 400
    silence_duration = 2.0
    max_record_time = 20

    silence_chunks_limit = int(silence_duration / 0.1)
    silence_counter = 0
    speaking_started = False
    recording = []

    start_time = time.time()

    with stream:
        while True:
            data = q.get()
            volume = np.abs(data).mean()

            if volume > silence_threshold:
                speaking_started = True
                silence_counter = 0
                recording.append(data)
            else:
                if speaking_started:
                    silence_counter += 1
                    recording.append(data)
                    if silence_counter > silence_chunks_limit:
                        break

            if time.time() - start_time > max_record_time:
                break

    if recording:
        audio = np.concatenate(recording, axis=0)
        sf.write(AUDIO_FILE, audio, SAMPLE_RATE)

    print("Recording stopped.")

# =========================
# TRANSCRIPTION
# =========================
def transcribe_audio():
    with open(AUDIO_FILE, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(AUDIO_FILE, file.read()),
            model="whisper-large-v3-turbo",
            temperature=0,
            response_format="verbose_json"
        )

    return transcription.text.strip().lower()

# =========================
# GENERAL LLM
# =========================
def ask_llama(question):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": question}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

# =========================
# DUCKDUCKGO SEARCH (FREE)
# =========================

def duckduckgo_search(query):

    url = "https://html.duckduckgo.com/html/"
    
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    data = {
        "q": query
    }

    response = requests.post(url, headers=headers, data=data)

    soup = BeautifulSoup(response.text, "html.parser")

    results = []

    for result in soup.select(".result__snippet"):
        text = result.get_text().strip()
        if text:
            results.append(text)

    return results[:5]


def duck_rag_answer(query):
    docs = duckduckgo_search(query)

    if not docs:
        return "I couldn't retrieve relevant web information."

    context = "\n\n".join(docs)

    prompt = f"""
Use the following web search results to answer the question accurately.

Search Results:
{context}

Question:
{query}

Provide a clear and factual answer.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content

# =========================
# PLANNER
# =========================
def plan_request(text):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """
Return ONLY JSON.

If question involves current events, latest, today, news, or real-time info:
{"mode":"duck_rag"}

If weather:
{"mode":"weather","city":"city_name"}

Otherwise:
{"mode":"general"}
"""
            },
            {"role": "user", "content": text}
        ],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)

# =========================
# WEATHER
# =========================
def get_weather(city):
    city = city.replace("?", "").strip()

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url).json()

    if "main" not in response:
        return f"Weather API error: {response}"

    temp = response["main"]["temp"]
    description = response["weather"][0]["description"]

    return f"The current temperature in {city} is {temp}°C with {description}."

# =========================
# MAIN LOOP
# =========================
# =========================
# MAIN LOOP (UPGRADED)
# =========================
print("Voice AI started...")

# deterministic live-data keywords
LIVE_KEYWORDS = [
    "current", "latest", "today", "now",
    "news", "price", "score", "weather",
    "temperature", "who is", "who's",
    "president", "prime minister",
    "ceo", "stock", "bitcoin", "time",
    "date", "year"
]

while True:

    # 1. RECORD AUDIO
    record_audio()

    # 2. TRANSCRIBE
    text = transcribe_audio()

    if not text:
        continue

    print("You said:", text)

    # 3. EXIT COMMAND
    if "sayonara" in text:
        print("Sayonara. Shutting down...")
        break

    # 4. SPEAK MODE CONTROL
    speak_mode = False
    if "arigato" in text:
        speak_mode = True
        text = text.replace("arigato", "").strip()

    # =========================
    # 5. ROUTING LOGIC (HYBRID)
    # =========================

    text_lower = text.lower()

    # PRIORITY 1: Weather detection
    if "weather" in text_lower or "temperature" in text_lower:
        plan = plan_request(text)

    # PRIORITY 2: Deterministic live-data routing
    elif any(keyword in text_lower for keyword in LIVE_KEYWORDS):
        plan = {"mode": "duck_rag"}

    # PRIORITY 3: Planner LLM fallback
    else:
        try:
            plan = plan_request(text)
        except:
            plan = {"mode": "general"}

    print("PLAN:", plan)

    # =========================
    # 6. EXECUTE PLAN
    # =========================

    if plan["mode"] == "weather":

        city = plan.get("city", "")

        if not city:
            answer = "Please specify a city."
        else:
            answer = get_weather(city)

    elif plan["mode"] == "duck_rag":

        answer = duck_rag_answer(text)

    elif plan["mode"] == "general":

        answer = ask_llama(text)

        # automatic fallback if LLM lacks live info
        if any(phrase in answer.lower() for phrase in [
            "don't have access to live",
            "knowledge cutoff",
            "cannot browse",
            "no real-time"
        ]):
            print("Falling back to DuckDuckGo...")
            answer = duck_rag_answer(text)

    else:

        answer = "I don't understand the request."

    # =========================
    # 7. OUTPUT
    # =========================

    print("\nAI:", answer)
    print("-" * 60)

    if speak_mode:
        speak(answer)


# =========================
# CLEANUP
# =========================
if os.path.exists(AUDIO_FILE):
    os.remove(AUDIO_FILE)

print("Program terminated cleanly.")



