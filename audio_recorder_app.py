import streamlit as st
import pyaudio
import wave
import os
import requests
import joblib
import whisper
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1000)

model=joblib.load('language_detection.sav')


def main():
    st.title("Audio Recorder")

    # User input for audio recording
    recording = st.checkbox("Start Recording")
    if recording:
        record_audio()
    
    st.title("MP3 File Uploader")
    st.write("Upload an MP3 file, and we'll process it!")
    mp3_file = st.file_uploader("Upload an MP3 file", type=["mp3"])
    if mp3_file:
        upload_audio(mp3_file)

def record_audio():
    frames = []
    audio_format = pyaudio.paInt16
    channels = 1
    sample_rate = 44100
    chunk_size = 1024
    duration = 10  # Maximum recording duration in seconds

    p = pyaudio.PyAudio()
    stream = p.open(format=audio_format, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk_size)

    st.info("Recording...")

    for _ in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)

    st.success("Recording complete!")

    # Save the recorded audio to a WAV file
    save_audio(frames)

    # Clean up resources
    stream.stop_stream()
    stream.close()
    p.terminate()
    
def upload_audio(mp3_file):
    if mp3_file is not None:
    # Check if the uploaded file is an MP3 file
        if mp3_file.type == "audio/mpeg":
            st.success(f"File '{mp3_file.name}' uploaded successfully!")
        
        temp_mp3_file_path = os.path.join("temp", mp3_file.name)

        os.makedirs("temp", exist_ok=True)
        with open(temp_mp3_file_path, "wb") as f:
            f.write(mp3_file.read())

    else:
        st.error("Please upload a valid MP3 file.")
    
    
       
    transcribe(temp_mp3_file_path)

def save_audio(frames):
    if not os.path.exists("recordings"):
        os.makedirs("recordings")

    output_filename = os.path.join("recordings", "recorded_audio.wav")

    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)
    wf.writeframes(b''.join(frames))
    wf.close()

    st.audio(output_filename, format="audio/wav")
    
    
    
def transcribe(audio):
    

    model = whisper.load_model("large")
    result = model.transcribe(audio)
    text=result['text']
    
    st.write(text)
    
    test_function(text)

def test_function(sentence):
  ps=PorterStemmer()
  corpus=[]
  languages={
      '_Gujarati':0,
      '_Hindi':1,
      '_Marathi':2,
      '_Tamil':3,
      '_Telugu':4

  }
  rev =re.sub("^[a-zA-Z]"," ",sentence)
  rev=rev.lower()
  rev=rev.split()
  rev=[ps.stem(word) for word in rev if set(stopwords.words())]
  rev=' '.join(rev)
  corpus.append(rev)

  rev=cv.transform([rev]).toarray()
  output=model.predict(rev)[0]


  keys=list(languages)
  values=list(languages.values())
  position=values.index(output)

  output=keys[position]
  print(output)

    

    

if __name__ == "__main__":
    main()
