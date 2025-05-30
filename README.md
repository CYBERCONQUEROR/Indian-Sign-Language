# Indian Sign Language Translator #

<Empowering communication for millions through AI-powered Indian Sign Language (ISL) recognition>
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🧩 **Problem Statement**

Millions of individuals in India with hearing and speech impairments face communication barriers daily. While American Sign Language (ASL) has been extensively researched and supported by technology, Indian Sign Language (ISL) lacks such development.
Our mission is to build a real-time, user-friendly ISL Translator that:
>Recognizes ISL gestures using a standard webcam (no gloves, Kinect, or special hardware)

>Bridges the gap between ISL users and non-signers

>Promotes ISL learning and inclusivity


----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🧠 **Approach & Solution**

![ISHARA Logo](https://github.com/CYBERCONQUEROR/Indian-Sign-Language/blob/main/ISHARA.png?raw=true)

Despite limited datasets and language variation across regions, we aim to:

>Collect and curate ISL gesture data from internet sources

>Apply state-of-the-art ML algorithms like CNN and LSTM for gesture classification

>Use computer vision techniques for real-time gesture detection

>Build an end-to-end system that supports sign-to-text/speech and text/speech-to-sign translation
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🧩 **Core Components**

Feature	Description

✋ Sign-to-Text/Speech	Real-time detection of hand gestures via webcam and conversion to text or audio

🗣️ Speech/Text-to-Sign	Converts spoken or typed language into ISL using animated sign representations

📚 Learn ISL	Visual reference of static gestures (A–Z, 1–9) to promote ISL literacy

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

✨ **Key Features**

🔤 Real-Time Gesture Detection with OpenCV and deep learning

🗣️ Sign Language to Text & Speech Conversion

📢 Text/Speech to ISL Gesture Animation

📚 Interactive ISL Learning Module

🌐 Multilingual Support using Google Translate API

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🛠️ **Tech Stack**


Frontend	: Python Tkinter for GUI

Computer Vision :	OpenCV, NumPy, PIL

Machine Learning :	TensorFlow, Keras (CNN, LSTM models)

Speech & Translation :	gTTS, Google Translate API

Gesture Encoding :	Custom hackathon_encoder.pkl, dikshita.pkl (not disclosed)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📸 **Screenshots**

📂 Please check the Demo folder in the codebase to view the UI and application demo.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🚀 **How to Run**

1. Clone the Repository

     git clone https://github.com/your-username/Indian-Sign-Language.git

     cd Indian-Sign-Language

2. Install Dependencies

     pip install -r requirements.txt

3. Run the Application

     python eng_to_hindi.py

     python test.py

     🖼 These scripts offer two different GUI experiences.

    ⚠️ Note: Due to file size and privacy limitations, trained model weights and label encoders are not uploaded. You won’t be able to run real-time detection without them. Please refer to the demo folder for 
    application visuals.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📘 **Usage Tips**

Use the tab navigation to switch between translation modes

For sign detection:

Ensure good lighting

Keep your hand within the webcam frame

Use a microphone or text input box for speech-to-sign translation

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🙏 **Acknowledgments**

This project is a small but significant step toward inclusive communication and digital accessibility for the hearing and speech impaired community in India.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📬 **Feedback & Contributions**

Feel free to fork, improve, or suggest features. Let's make ISL more accessible together!
