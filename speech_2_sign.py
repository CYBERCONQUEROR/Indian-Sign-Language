import tempfile
import tkinter as tk
from tkinter import messagebox
from urllib.parse import _NetlocResultMixinBase
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import pickle
from tensorflow.keras.models import load_model # type: ignore
import time
import threading
from difflib import get_close_matches
import speech_recognition as sr
import os
from pydub import AudioSegment

# # Load model and encoder
# model = load_model("sign_language_cnn_model_double_hand.h5")
# with open("double_hand_label_encoder.pkl", "rb") as f:
#     le = pickle.load(f)

model = load_model("Dikshita.h5")
with open("dikshita.pkl", "rb") as f:
    le = pickle.load(f)


# TTS
engine = pyttsx3.init()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Autocomplete suggestion list
WORDS = ["hello", "help", "hi", "how", "are", "you", "my", "name", "is", "thanks", "thank", "bye"]

# Normalize

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 2)
    base = landmarks[0]
    centered = landmarks - base
    max_value = np.max(np.linalg.norm(centered, axis=1))
    return (centered / max_value).reshape(42, 2, 1) if max_value != 0 else centered.reshape(42, 2, 1)

# Extract

def extract_landmarks_from_frame(frame):
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    landmark_list = None
    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) == 2:
            lms1 = [[lm.x, lm.y] for lm in results.multi_hand_landmarks[0].landmark]
            lms2 = [[lm.x, lm.y] for lm in results.multi_hand_landmarks[1].landmark]
        else:
            lms1 = [[lm.x, lm.y] for lm in results.multi_hand_landmarks[0].landmark]
            lms2 = [[0, 0]] * 21
        landmark_list = normalize_landmarks(lms1 + lms2)
    return landmark_list, results.multi_hand_landmarks

class SignRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Recognition")
        self.root.configure(bg="#121212")
        self.root.attributes('-fullscreen', True)

        self.cap = cv2.VideoCapture(0)
        self.final_sentence = ""
        self.current_word = ""
        self.prediction_stability = []
        self.blank_frame_counter = 0

        self.logo_screen()

    def logo_screen(self):
        self.logo_frame = tk.Frame(self.root, bg="#121212")
        self.logo_frame.pack(expand=True, fill=tk.BOTH)

        title = tk.Label(self.logo_frame, text="🖐️ SIGN LANGUAGE INTERPRETER", font=("Helvetica", 28, "bold"), fg="#00e0ff", bg="#121212")
        title.pack(pady=60)

        tk.Button(self.logo_frame, text="▶ Start Recognition", font=("Helvetica", 18), bg="#00adb5", fg="white", padx=20, pady=10, command=self.start_app).pack(pady=20)
        tk.Button(self.logo_frame, text="❌ Exit", font=("Helvetica", 18), bg="#ff4c4c", fg="white", padx=20, pady=10, command=self.root.quit).pack()

    def start_app(self):
        self.logo_frame.destroy()
        self.create_ui()
        self.update_frame()

    def create_ui(self):
        self.video_border = tk.Frame(self.root, bg="#00adb5", bd=10)
        self.video_border.pack(pady=20)

        self.video_label = tk.Label(self.video_border)
        self.video_label.pack()

        self.pred_label = tk.Label(self.root, text="Predicted: ", font=("Helvetica", 20), fg="#00ffcc", bg="#121212")
        self.pred_label.pack(pady=5)

        self.word_label = tk.Label(self.root, text="Current Word: ", font=("Helvetica", 16), fg="white", bg="#121212")
        self.word_label.pack()

        self.sentence_label = tk.Label(self.root, text="Sentence: ", font=("Helvetica", 16), fg="white", bg="#121212")
        self.sentence_label.pack()

        self.suggestion_label = tk.Label(self.root, text="Suggestions: ", font=("Helvetica", 14), fg="gray", bg="#121212")
        self.suggestion_label.pack()

        button_frame = tk.Frame(self.root, bg="#121212")
        button_frame.pack(pady=15)

        tk.Button(button_frame, text="🔤 Text", font=("Helvetica", 14), command=self.show_sentence, bg="#007acc", fg="white").pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="🔊 Speak", font=("Helvetica", 14), command=self.speak_sentence, bg="#3cb371", fg="white").pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="❎ Clear Word", font=("Helvetica", 14), command=self.clear_word, bg="#ffa500", fg="white").pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="🧹 Clear Sentence", font=("Helvetica", 14), command=self.clear_sentence, bg="#8a2be2", fg="white").pack(side=tk.LEFT, padx=10)
        # tk.Button(button_frame, text="🗣️ Talk to Them", font=("Helvetica", 14), command=self.talk_to_them, bg="#ff69b4", fg="white").pack(side=tk.LEFT, padx=10)
        # "Talk to Them" Button
        talk_button = tk.Button(self.root, text="🗣️ Talk to Them", command=self.talk_to_them, font=("Helvetica", 14), bg="#3b3b3b", fg="white", padx=10, pady=5)
        talk_button.pack(pady=10)

        tk.Button(button_frame, text="❌ Exit", font=("Helvetica", 14), command=self.exit_app, bg="#ff4c4c", fg="white").pack(side=tk.LEFT, padx=10)
        # tk.Button(button_frame, text="🔍 Show All Signs", font=("Helvetica", 14), command=self.show_all_signs_display, bg="#00adb5", fg="white").pack(side=tk.LEFT, padx=10 )
        # tk.Button(button_frame, text="🔤 Show Sign Images", font=("Helvetica", 14), command=self.display_sign_images, bg="#007acc", fg="white").pack(side=tk.LEFT, padx=10)
        

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        landmarks, results = extract_landmarks_from_frame(frame)

        if results:
            for handLms in results:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        if landmarks is not None:
            prediction = model.predict(np.expand_dims(landmarks, axis=0), verbose=0)
            label = le.inverse_transform([np.argmax(prediction)])[0]
            self.pred_label.config(text=f"Predicted: {label}")
            self.prediction_stability.append(label)

            if len(self.prediction_stability) > 15:
                most_common = max(set(self.prediction_stability), key=self.prediction_stability.count)
                if self.prediction_stability.count(most_common) > 10:
                    if len(self.current_word) == 0 or self.current_word[-1] != most_common:
                        self.current_word += most_common
                        self.word_label.config(text=f"Current Word: {self.current_word}")
                        self.show_suggestions()
                    self.prediction_stability.clear()
                    self.blank_frame_counter = 0
        else:
            self.blank_frame_counter += 1
            if self.blank_frame_counter >= 25 and self.current_word:
                self.final_sentence += self.current_word + " "
                self.sentence_label.config(text=f"Sentence: {self.final_sentence.strip()}")
                self.current_word = ""
                self.word_label.config(text="Current Word: ")
                self.suggestion_label.config(text="Suggestions: ")
                self.blank_frame_counter = 0
                self.prediction_stability.clear()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(img))
        self.video_label.configure(image=img)
        self.video_label.image = img

        self.root.after(15, self.update_frame)

    def show_suggestions(self):
        matches = get_close_matches(self.current_word.lower(), WORDS, n=3, cutoff=0.5)
        if matches:
            self.suggestion_label.config(text=f"Suggestions: {', '.join(matches)}")
            if self.current_word.lower() in matches:
                self.current_word = matches[0]
                self.word_label.config(text=f"Current Word: {self.current_word}")
        else:
            self.suggestion_label.config(text="Suggestions: (No match)")

    def show_sentence(self):
        messagebox.showinfo("Final Sentence", self.final_sentence.strip() or "No sentence yet.")

    def speak_sentence(self):
        text = self.final_sentence.strip()
        def speak():
            engine.say(text)
            engine.runAndWait()
        threading.Thread(target=speak).start()

    def clear_word(self):
        self.current_word = ""
        self.word_label.config(text="Current Word: ")
        self.suggestion_label.config(text="Suggestions: ")
        self.prediction_stability.clear()

    def clear_sentence(self):

        self.final_sentence = ""
        self.sentence_label.config(text="Sentence: ")
        self.current_word = ""
        self.word_label.config(text="Current Word: ")

    def talk_to_them(self):

        # === Toplevel sign window ===
        sign_window = tk.Toplevel(self.root)
        sign_window.title("Sign Language Display")
        sign_window.geometry("1000x400")
        sign_window.configure(bg="#1e1e1e")

        label = tk.Label(sign_window, text="Listening...", font=("Helvetica", 16, "bold"), fg="white", bg="#1e1e1e")
        label.pack(pady=10)

        # === Scrollable Canvas with Horizontal Scrollbar ===
        canvas_frame = tk.Frame(sign_window, bg="#1e1e1e")
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(canvas_frame, bg="#1e1e1e", highlightthickness=0)
        h_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=canvas.xview)
        canvas.configure(xscrollcommand=h_scrollbar.set)

        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll_frame = tk.Frame(canvas, bg="#1e1e1e")
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        scroll_frame.bind("<Configure>", on_configure)

        # === Display animation ===
        def display_words_with_animation(text):
            image_folder = r"C:\Users\Krishnav\OneDrive\Desktop\AKTU\Today"
            words = text.split()

            def display_word(word_index=0):
                if word_index >= len(words):
                    return

                word = words[word_index]
                letters = [char.upper() for char in word if char.isalpha()]

                word_frame = tk.Frame(scroll_frame, bg="#2c2c2c", padx=10, pady=10, bd=2, relief="solid")
                word_frame.pack(side=tk.LEFT, padx=12, pady=12)
                word_frame.configure(highlightbackground="#00FF99", highlightthickness=2)

                def animate_letter(index=0):
                    if index >= len(letters):
                        return
                    letter = letters[index]
                    img_path = os.path.join(image_folder, f"{letter}.jpg")
                    if os.path.exists(img_path):
                        img = Image.open(img_path).resize((70, 70))
                        photo = ImageTk.PhotoImage(img)

                        img_label = tk.Label(word_frame, image=photo, bg="#2c2c2c")
                        img_label.image = photo
                        img_label.pack(side=tk.LEFT, padx=2)

                    word_frame.after(100, lambda: animate_letter(index + 1))

                animate_letter()
                sign_window.after(1800, lambda: display_word(word_index + 1))

            display_word()

        # === Listening thread ===
        def listen_and_display():
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = True

            with sr.Microphone() as source:
                try:
                    label.config(text="🎤 Calibrating for background noise...")
                    recognizer.adjust_for_ambient_noise(source, duration=1)
                    label.config(text="🎤 Listening...")

                    audio = recognizer.listen(source, timeout=8, phrase_time_limit=7)

                    label.config(text="🔄 Recognizing...")
                    text = recognizer.recognize_google(audio)

                    label.config(text=f"🗣️ You said: {text}")
                    display_words_with_animation(text.lower())

                except sr.UnknownValueError:
                    label.config(text="❌ Couldn't understand. Try again.")
                except sr.RequestError:
                    label.config(text="⚠️ API error. Check internet.")
                except sr.WaitTimeoutError:
                    label.config(text="⌛ No speech detected.")
                except Exception as e:
                    label.config(text=f"❗ Error: {str(e)}")

        threading.Thread(target=listen_and_display).start()




    def exit_app(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignRecognizerApp(root)
    root.mainloop()
    root.destroy()
    cv2.destroyAllWindows()
