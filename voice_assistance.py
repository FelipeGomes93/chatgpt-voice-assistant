# ==============================
# 1. IMPORTS
# ==============================
import openai
import whisper
import os
from gtts import gTTS
from IPython.display import Audio, display, Javascript
from google.colab import output
from base64 import b64decode

# ==============================
# 2. CONFIGURAÇÕES
# ==============================
language = "pt"  # "pt" ou "en"
os.environ['OPENAI_API_KEY'] = 'SUA_API_KEY_AQUI'
openai.api_key = os.environ.get('OPENAI_API_KEY')

# ==============================
# 3. GRAVAÇÃO DE ÁUDIO
# ==============================
RECORD = """
const sleep  = time => new Promise(resolve => setTimeout(resolve, time))
const b2text = blob => new Promise(resolve => {
  const reader = new FileReader()
  reader.onloadend = e => resolve(e.srcElement.result)
  reader.readAsDataURL(blob)
})
var record = time => new Promise(async resolve => {
  stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  recorder = new MediaRecorder(stream)
  chunks = []
  recorder.ondataavailable = e => chunks.push(e.data)
  recorder.start()
  await sleep(time)
  recorder.onstop = async ()=>{
    blob = new Blob(chunks)
    text = await b2text(blob)
    resolve(text)
  }
  recorder.stop()
})
"""

def record(sec=5):
    display(Javascript(RECORD))
    js_result = output.eval_js(f'record({sec * 1000})')
    audio = b64decode(js_result.split(',')[1])
    file_name = 'request_audio.wav'
    with open(file_name, 'wb') as f:
        f.write(audio)
    return f'/content/{file_name}'

# ==============================
# 4. GRAVA ÁUDIO
# ==============================
print("🎤 Ouvindo...")
record_file = record(5)
display(Audio(record_file))

# ==============================
# 5. SPEECH → TEXT (Whisper)
# ==============================
model = whisper.load_model("small")
result = model.transcribe(record_file, fp16=False, language=language)
transcription = result["text"]

print("\n📝 Você disse:")
print(transcription)

# ==============================
# 6. CHATGPT
# ==============================
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": transcription}]
)

chatgpt_response = response.choices[0].message.content

print("\n🤖 Resposta:")
print(chatgpt_response)

# ==============================
# 7. TEXT → SPEECH (gTTS)
# ==============================
tts = gTTS(text=chatgpt_response, lang=language)
response_audio = "/content/response_audio.mp3"
tts.save(response_audio)

print("\n🔊 Respondendo em áudio...")
display(Audio(response_audio, autoplay=True))