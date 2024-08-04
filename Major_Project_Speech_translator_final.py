import streamlit as st
import threading
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
import io
import torch
import time
from queue import Queue

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("C:\\AI Major\\Major_Group_Project\\temp_fine-tuned-model")
model = AutoModelForSeq2SeqLM.from_pretrained("C:\\AI Major\\Major_Group_Project\\temp_fine-tuned-model")

def translate_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    with torch.no_grad():
        translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

def text_to_speech(text, lang='hi'):
    tts = gTTS(text=text, lang=lang)
    audio_file = io.BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    return audio_file

def process_audio_stream(queue):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=5)  # Adjust timeout as needed
                english_text = recognizer.recognize_google(audio)
                translated_text = translate_text(english_text, tokenizer, model)
                audio_file = text_to_speech(translated_text)
                
                # Put results in the queue
                queue.put({
                    "english_text": english_text,
                    "translated_text": translated_text,
                    "audio_file": audio_file
                })
        except sr.UnknownValueError:
            continue
        except sr.RequestError:
            continue
        except Exception as e:
            print(f"An error occurred: {e}")
        time.sleep(1)  # Polling interval

def main():
    st.title("Real-Time Speech Translator")

    # Create a queue to share data between threads
    queue = Queue()

    # Start audio processing in a separate thread
    threading.Thread(target=process_audio_stream, args=(queue,), daemon=True).start()

    # Display results
    st.write("Listening for audio...")
    
    # Streamlit loop to keep the app responsive
    while True:
        if not queue.empty():
            result = queue.get()
            st.write("Recognized Text (English):", result["english_text"])
            st.write("Translated Text (Hindi):", result["translated_text"])
            st.audio(result["audio_file"], format='audio/mp3')

        time.sleep(1)  # Update interval for Streamlit

if __name__ == "__main__":
    main()
