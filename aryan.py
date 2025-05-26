import customtkinter as ctk # Use customtkinter
from tkinter import messagebox # Keep standard messagebox for simplicity
# import tempfile # Not used
# from urllib.parse import _NetlocResultMixinBase # Not used
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
# from pydub import AudioSegment # Not used in provided code
from googletrans import Translator # Keep original googletrans import
import tkinter as tk # Keep tk import specifically for tk.Canvas

# --- Load Model and Encoder ---
MODEL_LOADED = False
TTS_ENABLED = False
MEDIAPIPE_LOADED = False
LE_LOADED = False

try:
    model = load_model("sign_language_cnn_model_double_hand.h5")
    MODEL_LOADED = True
    with open("double_hand_label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    LE_LOADED = True
except Exception as e:
    print(f"Error loading model or encoder: {e}")
    # Optionally show error in UI later

# --- Text-to-Speech Setup ---
try:
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    if len(voices) > 1:
        try:
            engine.setProperty('voice', voices[1].id) # Try female voice
        except IndexError:
            engine.setProperty('voice', voices[0].id) # Fallback to default
    elif voices:
         engine.setProperty('voice', voices[0].id)
    else:
        print("No pyttsx3 voices found.")
    engine.setProperty('rate', 160)
    engine.setProperty('volume', 1.0)
    TTS_ENABLED = True
except Exception as e:
    print(f"Error initializing pyttsx3: {e}")

# --- MediaPipe Setup ---
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    MEDIAPIPE_LOADED = True
except Exception as e:
    print(f"Error initializing MediaPipe: {e}")

# --- Autocomplete Suggestion List ---
WORDS = ["hello", "help", "hi", "how", "are", "you", "my", "name", "is", "thanks", "thank", "bye", "good", "morning", "afternoon", "evening", "night", "please", "yes", "no", "maybe"]

# --- Landmark Normalization (Original) ---
def normalize_landmarks(landmarks):
    landmarks_np = np.array(landmarks).reshape(-1, 2)
    base = landmarks_np[0]
    centered = landmarks_np - base
    max_value = np.max(np.linalg.norm(centered, axis=1))
    # Ensure reshape matches model input (original logic used 42,2,1)
    return (centered / max_value).reshape(42, 2, 1) if max_value != 0 else centered.reshape(42, 2, 1)

# --- Landmark Extraction (Original - slight modification for consistency) ---
def extract_landmarks_from_frame(frame):
    landmark_list = None
    multi_hand_landmarks_results = None
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    multi_hand_landmarks_results = results.multi_hand_landmarks

    if multi_hand_landmarks_results:
        all_landmarks = []
        hand_count = len(multi_hand_landmarks_results)

        if hand_count == 1:
            lms1 = [[lm.x, lm.y] for lm in multi_hand_landmarks_results[0].landmark]
            lms2 = [[0.0, 0.0]] * 21 # Zero padding
            # Consistent ordering attempt (simple version)
            if multi_hand_landmarks_results[0].landmark[0].x < 0.5:
                 all_landmarks = lms1 + lms2
            else:
                 all_landmarks = lms2 + lms1
        elif hand_count == 2:
            lms1 = [[lm.x, lm.y] for lm in multi_hand_landmarks_results[0].landmark]
            lms2 = [[lm.x, lm.y] for lm in multi_hand_landmarks_results[1].landmark]
            # Consistent ordering attempt
            if multi_hand_landmarks_results[0].landmark[0].x < multi_hand_landmarks_results[1].landmark[0].x:
                all_landmarks = lms1 + lms2
            else:
                all_landmarks = lms2 + lms1

        if all_landmarks:
             # Check if we have exactly 42 points (21 landmarks * 2 hands) before normalizing
             if len(all_landmarks) == 42:
                 try:
                    landmark_list = normalize_landmarks(all_landmarks)
                 except Exception as e:
                    print(f"Error normalizing landmarks: {e}") # Catch potential errors here
             else:
                print(f"Warning: Expected 42 landmarks, got {len(all_landmarks)}")


    return landmark_list, multi_hand_landmarks_results

# --- Main Application Class ---
class SignRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Interpreter")
        # self.root.attributes('-fullscreen', True)
        self.root.state('zoomed')

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # --- CustomTkinter Settings ---
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.root.configure(fg_color="#121212")

        # --- State Variables (Original) ---
        self.cap = None
        self.final_sentence = ""
        self.current_word = ""
        self.prediction_stability = []
        self.blank_frame_counter = 0
        self.is_running = True # Flag for loops

        # --- Check Dependencies Needed for UI ---
        if not MEDIAPIPE_LOADED or not LE_LOADED or not MODEL_LOADED:
            self.show_error_screen()
        else:
            self.logo_screen()

    def show_error_screen(self):
        # Use CTk widgets for error screen
        self.error_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.error_frame.pack(expand=True, fill=tk.BOTH, padx=50, pady=50)

        error_message = "Error initializing critical components.\n\n"
        if not MODEL_LOADED: error_message += "- Failed to load Keras model.\n"
        if not LE_LOADED: error_message += "- Failed to load label encoder.\n"
        if not MEDIAPIPE_LOADED: error_message += "- Failed to initialize MediaPipe.\n"
        error_message += "\nPlease check console, file paths, and dependencies."

        title = ctk.CTkLabel(self.error_frame, text="⚠️ Initialization Error",
                              font=ctk.CTkFont(size=28, weight="bold"), text_color="#FF6347")
        title.pack(pady=(60, 20))
        details = ctk.CTkLabel(self.error_frame, text=error_message,
                              font=ctk.CTkFont(size=16), text_color="white", justify=tk.LEFT)
        details.pack(pady=20)
        ctk.CTkButton(self.error_frame, text="❌ Exit", font=ctk.CTkFont(size=18),
                      fg_color="#ff4c4c", hover_color="#E04040", command=self.root.quit,
                      width=150, height=40, corner_radius=8).pack(pady=30)

    def logo_screen(self):
        # Use CTk widgets for logo screen
        self.logo_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.logo_frame.pack(expand=True, fill=tk.BOTH, padx=50, pady=50)

        title = ctk.CTkLabel(self.logo_frame, text="🖐️ SIGN LANGUAGE INTERPRETER",
                              font=ctk.CTkFont(size=36, weight="bold"), text_color="#00E0FF")
        title.pack(pady=(80, 40))

        button_container = ctk.CTkFrame(self.logo_frame, fg_color="transparent")
        button_container.pack(pady=20)

        ctk.CTkButton(button_container, text="▶ Start Recognition", font=ctk.CTkFont(size=20),
                      command=self.start_app, width=250, height=50,
                      fg_color="#00adb5", hover_color="#007A7F", corner_radius=10).pack(pady=15)
        ctk.CTkButton(button_container, text="❌ Exit", font=ctk.CTkFont(size=20),
                      command=self.exit_app, width=250, height=50, # Use exit_app for proper shutdown
                      fg_color="#ff4c4c", hover_color="#E04040", corner_radius=10).pack(pady=15)

    def start_app(self):
        self.logo_frame.destroy()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam.")
            self.exit_app()
            return
        self.create_ui()
        self.update_frame() # Start the main loop

    def create_ui(self):
        # Main container frame using CTk
        main_frame = ctk.CTkFrame(self.root, fg_color="#1E1E1E")
        main_frame.pack(pady=20, padx=20, fill="both", expand=True)
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)

        # Video Area (Left)
        video_container = ctk.CTkFrame(main_frame, fg_color="#2c2c2c", corner_radius=10)
        video_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        video_title = ctk.CTkLabel(video_container, text="Camera Feed",
                                font=ctk.CTkFont(size=16, weight="bold"), text_color="#00E0FF")
        video_title.pack(pady=(5, 5))
        self.video_label = ctk.CTkLabel(video_container, text="")  # CTkLabel for video display
        self.video_label.pack(pady=5, padx=5, fill="both", expand=True)

        # Controls Area (Right)
        controls_container = ctk.CTkFrame(main_frame, fg_color="transparent", width=350)
        controls_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        controls_container.pack_propagate(False)

        # Labels
        self.pred_label = ctk.CTkLabel(controls_container, text="Predicted: --",
                                    font=ctk.CTkFont(size=24, weight="bold"), text_color="#00FFCC", wraplength=330)
        self.pred_label.pack(pady=15, padx=10, anchor="w")
        self.word_label = ctk.CTkLabel(controls_container, text="Current Word: ",
                                    font=ctk.CTkFont(size=18), text_color="white", wraplength=330)
        self.word_label.pack(pady=10, padx=10, anchor="w")
        self.sentence_label = ctk.CTkLabel(controls_container, text="Sentence: ",
                                        font=ctk.CTkFont(size=18), text_color="white", wraplength=330)
        self.sentence_label.pack(pady=10, padx=10, anchor="w")
        self.suggestion_label = ctk.CTkLabel(controls_container, text="Suggestions: ",
                                            font=ctk.CTkFont(size=16), text_color="gray", wraplength=330)
        self.suggestion_label.pack(pady=10, padx=10, anchor="w")

        # Buttons Frame
        button_frame = ctk.CTkFrame(controls_container, fg_color="transparent")
        button_frame.pack(pady=20, padx=10, fill="x")

        # Button Style
        button_font = ctk.CTkFont(size=14)
        button_width = 150
        button_height = 35
        button_corner_radius = 8
        button_pady = 5

        # Buttons Grid
        ctk.CTkButton(button_frame, text="🔤 Text", font=button_font, command=self.show_sentence,
                    width=button_width, height=button_height, corner_radius=button_corner_radius,
                    fg_color="#007acc", hover_color="#005C99").grid(row=0, column=0, padx=5, pady=button_pady)
        ctk.CTkButton(button_frame, text="🔊 Speak", font=button_font, command=self.speak_sentence,
                    width=button_width, height=button_height, corner_radius=button_corner_radius,
                    fg_color="#3cb371", hover_color="#2E8B57",
                    state="normal" if TTS_ENABLED else "disabled").grid(row=0, column=1, padx=5, pady=button_pady)
        ctk.CTkButton(button_frame, text="❎ Clear Word", font=button_font, command=self.clear_word,
                    width=button_width, height=button_height, corner_radius=button_corner_radius,
                    fg_color="#ffa500", hover_color="#CC8400").grid(row=1, column=0, padx=5, pady=button_pady)
        ctk.CTkButton(button_frame, text="🧹 Clear Sentence", font=button_font, command=self.clear_sentence,
                    width=button_width, height=button_height, corner_radius=button_corner_radius,
                    fg_color="#8a2be2", hover_color="#6B20B0").grid(row=1, column=1, padx=5, pady=button_pady)
        ctk.CTkButton(button_frame, text="🗣️ Talk to Them", font=button_font, command=self.talk_to_them,
                    width=button_width, height=button_height, corner_radius=button_corner_radius,
                    fg_color="#555555", hover_color="#3b3b3b").grid(row=2, column=0, padx=5, pady=button_pady)
        ctk.CTkButton(button_frame, text="👀 Show Signs", font=button_font, command=self.show_gestures,
                    width=button_width, height=button_height, corner_radius=button_corner_radius,
                    fg_color="#ff6347", hover_color="#E05030").grid(row=2, column=1, padx=5, pady=button_pady)

        # Exit Button
        ctk.CTkButton(button_frame, text="❌ Exit App", font=button_font, command=self.exit_app,
                    width=button_width*2 + 10, height=button_height, corner_radius=button_corner_radius,
                    fg_color="#ff4c4c", hover_color="#E04040").grid(row=3, column=0, columnspan=2, padx=5, pady=(20, 5))

    # --- update_frame: Original Logic ---
    def update_frame(self):
        if not self.is_running or not self.cap or not self.cap.isOpened():
             return

        ret, frame = self.cap.read()
        if not ret:
            # Optionally add a small delay before trying again or log error
            if self.is_running:
                 self.root.after(50, self.update_frame) # Try again shortly
            return

        frame = cv2.flip(frame, 1)
        frame_copy_for_drawing = frame.copy() # Draw landmarks on a copy

        landmarks, hand_landmarks_results = extract_landmarks_from_frame(frame)

        # Draw Landmarks if detected
        if hand_landmarks_results:
            for handLms in hand_landmarks_results:
                mp_draw.draw_landmarks(frame_copy_for_drawing, handLms, mp_hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1), # Cyan dots
                                       mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1)) # Green lines

        # --- Original Prediction Logic ---
        if landmarks is not None:
            try:
                prediction = model.predict(np.expand_dims(landmarks, axis=0), verbose=0)
                label = le.inverse_transform([np.argmax(prediction)])[0]
                self.pred_label.configure(text=f"Predicted: {label}")
                self.prediction_stability.append(label)
                self.blank_frame_counter = 0 # Reset counter if prediction occurs

                # --- Original Stability & Word Building Logic ---
                if len(self.prediction_stability) > 15:
                    most_common = max(set(self.prediction_stability), key=self.prediction_stability.count)
                    if self.prediction_stability.count(most_common) > 10: # Stability threshold
                        if not self.current_word or self.current_word[-1] != most_common:
                             self.current_word += most_common
                             self.word_label.configure(text=f"Current Word: {self.current_word}")
                             self.show_suggestions()
                        # Clear buffer AFTER finding stable prediction and adding char (if needed)
                        self.prediction_stability.clear()

            except Exception as e:
                 print(f"Error during prediction or label transform: {e}")
                 # Optionally clear stability buffer or handle error state
                 self.prediction_stability.clear()

        else: # No landmarks detected
            self.blank_frame_counter += 1
            # --- Original Sentence Building on Blank Frames ---
            if self.blank_frame_counter >= 25 and self.current_word: # Blank frame threshold
                self.final_sentence += self.current_word + " "
                self.sentence_label.configure(text=f"Sentence: {self.final_sentence.strip()}")
                self.current_word = ""
                self.word_label.configure(text="Current Word: ")
                self.suggestion_label.configure(text="Suggestions: ")
                self.blank_frame_counter = 0 # Reset counter after adding word
                self.prediction_stability.clear() # Clear buffer when word is added


        # --- Display Frame using CTkLabel ---
        try:
            label_w = self.video_label.winfo_width()
            label_h = self.video_label.winfo_height()
            if label_w > 10 and label_h > 10:
                 frame_h, frame_w, _ = frame_copy_for_drawing.shape
                 aspect_ratio = frame_w / frame_h
                 new_w = label_w
                 new_h = int(new_w / aspect_ratio)
                 if new_h > label_h:
                     new_h = label_h
                     new_w = int(new_h * aspect_ratio)
                 if new_w != frame_w or new_h != frame_h:
                     frame_resized = cv2.resize(frame_copy_for_drawing, (new_w, new_h), interpolation=cv2.INTER_AREA)
                 else:
                    frame_resized = frame_copy_for_drawing
            else:
                frame_resized = frame_copy_for_drawing

            cv2image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height)) # Use CTkImage

            self.video_label.configure(image=ctk_img, text="") # Set image and clear text
            # self.video_label.image = ctk_img # Keep reference (might not be needed for CTkImage)

        except Exception as e:
            print(f"Error updating video label: {e}")


        # Schedule next update (original interval)
        if self.is_running:
            self.root.after(15, self.update_frame)

    # --- Original show_suggestions ---
    def show_suggestions(self):
        if not self.current_word:
             self.suggestion_label.configure(text="Suggestions: ")
             return
        matches = get_close_matches(self.current_word.lower(), WORDS, n=3, cutoff=0.5)
        if matches:
            self.suggestion_label.configure(text=f"Suggestions: {', '.join(matches)}")
            # Original auto-complete logic (kept as is)
            if self.current_word.lower() in matches:
                 # Check if the *first* match is the word itself - careful here
                 if self.current_word.lower() == matches[0]:
                     # Only update if it changed case, maybe? Avoid infinite loops.
                     if self.current_word != matches[0]:
                        self.current_word = matches[0]
                        self.word_label.configure(text=f"Current Word: {self.current_word}")
                 # Else, maybe just display suggestions without auto-correcting?
                 # The original code auto-corrected if current word was IN the matches list.
                 # Let's stick to the original logic for now, which might be overly aggressive:
                 # self.current_word = matches[0] # This was the implicit effect
                 # self.word_label.configure(text=f"Current Word: {self.current_word}")
        else:
            self.suggestion_label.configure(text="Suggestions: (No match)")

    # --- Original show_sentence ---
    def show_sentence(self):
        messagebox.showinfo("Final Sentence", self.final_sentence.strip() or "No sentence yet.")

    # --- Original speak_sentence ---
    def speak_sentence(self):
        text = self.final_sentence.strip()
        if not text:
            messagebox.showwarning("Speak", "Sentence is empty.")
            return
        if not TTS_ENABLED:
             messagebox.showerror("Speak Error", "Text-to-Speech engine not available.")
             return
        def speak():
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"Error during speech: {e}")
        threading.Thread(target=speak, daemon=True).start()

    # --- Original clear_word ---
    def clear_word(self):
        self.current_word = ""
        self.word_label.configure(text="Current Word: ")
        self.suggestion_label.configure(text="Suggestions: ")
        self.prediction_stability.clear()

    # --- Original clear_sentence ---
    def clear_sentence(self):
        self.final_sentence = ""
        self.sentence_label.configure(text="Sentence: ")
        self.clear_word() # Also clear the current word

    # --- talk_to_them: Enhanced UI, Original Logic ---
    def talk_to_them(self):
        # Use CTkToplevel
        sign_window = ctk.CTkToplevel(self.root)
        sign_window.title("Sign Language Display")
        sign_window.geometry("1000x450")
        sign_window.configure(fg_color="#1e1e1e")
        sign_window.transient(self.root)
        sign_window.grab_set()

        # Use CTkLabel for status
        status_label = ctk.CTkLabel(sign_window, text="Initializing...",
                                    font=ctk.CTkFont(size=16, weight="bold"), text_color="white")
        status_label.pack(pady=10)

        # Scrollable Canvas Area using CTkFrame, tk.Canvas, CTkScrollbar
        canvas_frame = ctk.CTkFrame(sign_window, fg_color="#1e1e1e")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Need tk.Canvas for compatibility with placing images directly in older code style
        canvas = tk.Canvas(canvas_frame, bg="#1e1e1e", highlightthickness=0)
        h_scrollbar = ctk.CTkScrollbar(canvas_frame, orientation=tk.HORIZONTAL, command=canvas.xview,
                                        button_color="#555555", button_hover_color="#777777")
        canvas.configure(xscrollcommand=h_scrollbar.set)

        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X, padx=2)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Inner frame using CTkFrame
        scroll_frame = ctk.CTkFrame(canvas, fg_color="#1e1e1e")
        canvas_window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        # Configure scroll region updates
        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        scroll_frame.bind("<Configure>", on_frame_configure)
        def update_canvas_scroll(event):
             canvas_width = event.width
             canvas.itemconfig(canvas_window, width=canvas_width)
             canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.bind('<Configure>', update_canvas_scroll)


        # --- Original display_words_with_animation ---
        # (Modified to use CTkFrame for word frames and CTkLabel for image placeholders)
        def display_words_with_animation(text):
            image_folder = r"F:\ISL\Today" # ** Check/Update this path **
            words = text.lower().split()
            current_x_pos = 10

            for widget in scroll_frame.winfo_children(): widget.destroy() # Clear previous

            if not os.path.isdir(image_folder):
                status_label.configure(text=f"❌ Error: Image folder not found:\n{image_folder}", text_color="#FF6347")
                return

            def display_next_word(word_index=0, current_x=10):
                if word_index >= len(words):
                    status_label.configure(text="✅ Display Finished", text_color="#3cb371")
                    sign_window.after(100, lambda: canvas.configure(scrollregion=canvas.bbox("all")))
                    sign_window.after(150, lambda: canvas.xview_moveto(1.0))
                    return

                word = words[word_index]
                letters = [char.upper() for char in word if char.isalnum()]
                if not letters:
                     display_next_word(word_index + 1, current_x)
                     return

                # Use CTkFrame for the word frame
                word_frame = ctk.CTkFrame(scroll_frame, fg_color="#2c2c2c",
                                           border_width=2, border_color="#00FF99", corner_radius=8)
                word_frame.place(x=current_x, y=10)

                word_width = 0
                letter_widgets = []

                # Use CTkLabel as placeholder/image holder
                for i, letter in enumerate(letters):
                    img_path = os.path.join(image_folder, f"{letter}.jpg")
                    img_label = ctk.CTkLabel(word_frame, text=f"{letter}?", text_color="gray", width=70, height=70) # Placeholder
                    img_label.pack(side=tk.LEFT, padx=3, pady=3)
                    letter_widgets.append((img_label, img_path))
                    word_width += 70 + 6

                word_frame.configure(width=word_width)

                # Load images (same logic, but configure CTkLabel)
                for label, path in letter_widgets:
                     if os.path.exists(path):
                         try:
                             img = Image.open(path).resize((70, 70), Image.Resampling.LANCZOS)
                             # Use CTkImage for better theme handling
                             ctk_photo = ctk.CTkImage(light_image=img, dark_image=img, size=(70,70))
                             label.configure(image=ctk_photo, text="")
                             # label.image = ctk_photo # Keep reference if needed
                         except Exception as e:
                             print(f"Error loading image {path}: {e}")
                             label.configure(text=f"{letter}\nError", text_color="red", wraplength=60, image=None) # Clear image on error
                     else:
                          print(f"Image not found: {path}")
                          label.configure(text=f"{letter}\nN/A", text_color="orange", image=None) # Clear image if not found

                next_x = current_x + word_width + 20
                delay_ms = len(letters) * 150 + 800 # Original delay logic
                sign_window.after(delay_ms, lambda wi=word_index + 1, nx=next_x: display_next_word(wi, nx))
                sign_window.after(100, lambda x_scroll=next_x: canvas.xview_moveto( x_scroll / max(1,scroll_frame.winfo_width()) ))

            display_next_word()

        # --- Original listen_and_display thread logic ---
        # (Uses googletrans, configure CTkLabel for status)
        def listen_and_display():
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 300 # Original value
            recognizer.dynamic_energy_threshold = True
            try:
                translator = Translator() # Original Translator init
                translator_ok = True
            except Exception as e:
                print(f"Failed to initialize googletrans Translator: {e}")
                # This is where the httpcore error would likely happen
                status_label.configure(text=f"❗ Error: Translator init failed!\n Check dependencies/internet.\n{e}", text_color="#FF6347", wraplength=900)
                translator_ok = False
                # Add close button immediately if translator fails
                close_btn = ctk.CTkButton(sign_window, text="Close", command=sign_window.destroy, width=100)
                close_btn.pack(pady=10)
                sign_window.grab_release()
                return # Stop the thread if translator fails

            with sr.Microphone() as source:
                try:
                    status_label.configure(text="🎤 Calibrating...")
                    sign_window.update_idletasks()
                    recognizer.adjust_for_ambient_noise(source, duration=1)

                    status_label.configure(text="🎤 Listening...")
                    sign_window.update_idletasks()
                    audio = recognizer.listen(source, timeout=8, phrase_time_limit=7) # Original timeouts

                    status_label.configure(text="🔄 Recognizing...")
                    sign_window.update_idletasks()
                    # --- Original Recognition & Translation ---
                    text_hi = recognizer.recognize_google(audio, language='hi-IN') # Tries Hindi first

                    status_label.configure(text=f"🗣️ Heard (HI): {text_hi}\nTranslating...", text_color="white")
                    sign_window.update_idletasks()

                    # --- Original Translation Call ---
                    translated = translator.translate(text_hi, src='hi', dest='en')
                    text_en = translated.text

                    status_label.configure(text=f"🗣️ Heard (HI): {text_hi}\nTranslated (EN): {text_en}", wraplength=900)

                    time.sleep(1)
                    display_words_with_animation(text_en) # Use translated text

                # --- Original Error Handling ---
                except sr.UnknownValueError:
                     try:
                         # Try recognizing in English if Hindi fails
                         status_label.configure(text="🔄 Recognizing (EN)...", text_color="orange")
                         sign_window.update_idletasks()
                         text_en = recognizer.recognize_google(audio, language='en-US')
                         status_label.configure(text=f"🗣️ Heard (EN): {text_en}", text_color="white")
                         time.sleep(1)
                         display_words_with_animation(text_en) # Display English directly
                     except sr.UnknownValueError:
                         status_label.configure(text="❌ Couldn't understand audio. Try again.", text_color="#FF6347")
                     except sr.RequestError as e_recog:
                          status_label.configure(text=f"⚠️ Recognition API error: {e_recog}", text_color="#FF6347")

                except sr.RequestError as e:
                     status_label.configure(text=f"⚠️ API error (Translate/Recognize): {e}", text_color="#FF6347")
                except sr.WaitTimeoutError:
                     status_label.configure(text="⌛ No speech detected.", text_color="yellow")
                except AttributeError as ae: # Catch the specific httpcore error if it happens here
                     status_label.configure(text=f"❗ Translator Error:\n{ae}\nCheck googletrans/httpcore versions.", text_color="#FF6347", wraplength=900)
                     print(f"Translator AttributeError: {ae}")
                except Exception as e:
                     status_label.configure(text=f"❗ Error: {str(e)}", text_color="#FF6347", wraplength=900)
                     print(f"Error in listen_and_display: {e}")

            # Add close button after finishing
            close_button = ctk.CTkButton(sign_window, text="Close", command=sign_window.destroy, width=100)
            close_button.pack(pady=10)
            sign_window.grab_release()

        # Start the original listening thread
        threading.Thread(target=listen_and_display, daemon=True).start()


    # --- show_gestures: Enhanced UI, Original Logic ---
    def show_gestures(self):
        # Use CTkToplevel
        gestures_window = ctk.CTkToplevel(self.root)
        gestures_window.title("Gesture Images (A-Z & 0-9)")
        gestures_window.geometry("900x700")
        gestures_window.configure(fg_color="#1E1E1E")
        gestures_window.transient(self.root)
        gestures_window.grab_set()

        title_label = ctk.CTkLabel(gestures_window, text="Available Sign Images",
                                   font=ctk.CTkFont(size=20, weight="bold"), text_color="#00E0FF")
        title_label.pack(pady=15)

        gesture_chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        image_folder = r"F:\ISL\Today" # ** Check/Update this path **

        if not os.path.isdir(image_folder):
             ctk.CTkLabel(gestures_window, text=f"Error: Image folder not found:\n{image_folder}",
                          font=ctk.CTkFont(size=16), text_color="#FF6347").pack(pady=30)
             ctk.CTkButton(gestures_window, text="Close", command=gestures_window.destroy).pack(pady=10)
             return

        # Use CTkScrollableFrame for easier implementation
        scroll_frame_container = ctk.CTkFrame(gestures_window, fg_color="transparent")
        scroll_frame_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        scroll_frame = ctk.CTkScrollableFrame(scroll_frame_container, fg_color="#2c2c2c",
                                               label_text="Signs", label_text_color="white")
        scroll_frame.pack(fill=tk.BOTH, expand=True)

        # Display images using CTk widgets in a grid
        items_per_row = 8
        image_size = (80, 80)
        loaded_images = {} # Simple caching

        for i, char in enumerate(gesture_chars):
            img_path = os.path.join(image_folder, f"{char}.jpg")

            # Use CTkFrame for each gesture item
            gesture_frame = ctk.CTkFrame(scroll_frame, fg_color="#3a3a3a", corner_radius=8)
            gesture_frame.grid(row=i // items_per_row, column=i % items_per_row, padx=10, pady=10)

            # Use CTkLabel for image
            img_label = ctk.CTkLabel(gesture_frame, text="?", text_color="gray", width=image_size[0], height=image_size[1])
            img_label.pack(pady=(5,0))

            # Use CTkLabel for character text
            label = ctk.CTkLabel(gesture_frame, text=char, font=ctk.CTkFont(size=14, weight="bold"), text_color="white")
            label.pack(pady=(0, 5))

            if os.path.exists(img_path):
                try:
                    if char not in loaded_images:
                        img = Image.open(img_path).resize(image_size, Image.Resampling.LANCZOS)
                        ctk_photo = ctk.CTkImage(light_image=img, dark_image=img, size=image_size)
                        loaded_images[char] = ctk_photo
                    else:
                         ctk_photo = loaded_images[char]
                    img_label.configure(image=ctk_photo, text="")
                    # img_label.image = ctk_photo # Keep reference if needed

                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    img_label.configure(text="Error", text_color="red", image=None)
            else:
                print(f"Image not found: {img_path}")
                img_label.configure(text="N/A", text_color="orange", image=None)

        # Close button using CTk
        close_button = ctk.CTkButton(gestures_window, text="Close", command=gestures_window.destroy, width=120)
        close_button.pack(pady=10)


    # --- Original exit_app ---
    def exit_app(self):
        print("Exiting application...")
        self.is_running = False # Stop update loop flag
        if self.cap:
            self.cap.release()
            print("Camera released.")
        self.root.destroy() # Destroy the main window
        print("Application closed.")


# --- Main Execution ---
if __name__ == "__main__":
    # Check if critical components loaded before creating window
    if not MEDIAPIPE_LOADED or not LE_LOADED or not MODEL_LOADED:
         print("\n--- Cannot start UI due to initialization errors (see details above) ---")
         # Optionally create a minimal error window if Tk can still run
         try:
             root = ctk.CTk()
             root.title("Initialization Error")
             root.geometry("400x200")
             app = SignRecognizerApp(root) # This will call show_error_screen
             root.mainloop()
         except Exception as tk_e:
             print(f"Failed to create even error window: {tk_e}")
    else:
        root = ctk.CTk() # Use CTk main window
        app = SignRecognizerApp(root)
        root.mainloop()
        # Cleanup happens in exit_app
        cv2.destroyAllWindows() # Ensure OpenCV windows are closed if any were opened separately