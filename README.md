Voice Biometric System

This project implements a simple voice recognition system using MFCC (Mel Frequency Cepstral Coefficients) features. It allows users to register their voice and later identify speakers based on their unique vocal characteristics.

Features
User Registration: Capture a user's voice and extract MFCC features for future identification.
Speaker Identification: Record a new voice sample and compare it with registered users to determine the closest match.
Distance-Based Matching: Uses numerical distance (e.g., Euclidean) between MFCC features to identify speakers.
Simple Command-Line Interface: Interactively register new users or identify speakers.
Folder Structure
voice-biometrics/



│
├─ users/                  # Stores registered users' audio and MFCC features

│   └─ <username>/         # Each user has a dedicated folder


│       ├─ voice.wav       # Recorded voice sample

│       └─ mfcc.npy        # Extracted MFCC features

│
├─ record.py               # Module for recording voice

├─ features.py             # Module for extracting and saving MFCC features

├─ compare.py              # Module for comparing MFCC features

└─ main.py                 # Main script with CLI for registration and identification


How It Works
Register a User
The system records the user's voice.
Extracts MFCC features from the recording.
Saves the features for future identification.
Identify a Speaker
The system records a voice sample.
Extracts MFCC features from the sample.
Computes distances to all registered users' MFCCs.
Identifies the speaker if the distance is below a defined threshold.
Usage
Run the system:
python main.py

Choose from the menu:
1 to register a new user.
2 to identify a speaker.
3 to exit.
Requirements
Python 3.x
Libraries: numpy, sounddevice (or equivalent for recording), scipy (optional for audio processing), librosa (for MFCC extraction)
