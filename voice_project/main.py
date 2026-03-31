import os
import numpy as np
from record import record_voice
from features import extract_and_save_mfcc
from compare import compare_mfcc

USERS_FOLDER = "users"
THRESHOLD = 15000

def ensure_user_folder(name):
    user_path = os.path.join(USERS_FOLDER, name)
    os.makedirs(user_path, exist_ok=True)
    return user_path

def register_user(name: str):
    user_path = ensure_user_folder(name)
    wav_path = os.path.join(user_path, "voice.wav")
    mfcc_path = os.path.join(user_path, "mfcc.npy")
    print(f"Registering user: {name}")
    record_voice(wav_path)
    extract_and_save_mfcc(wav_path, mfcc_path)
    print(f"{name} registered successfully.")

def identify_user():
    temp_wav = "temp.wav"
    temp_mfcc = "temp.npy"
    record_voice(temp_wav)
    test_mfcc = extract_and_save_mfcc(temp_wav, temp_mfcc)

    distances = {}
    for user in os.listdir(USERS_FOLDER):
        mfcc_file = os.path.join(USERS_FOLDER, user, "mfcc.npy")
        if not os.path.exists(mfcc_file):
            continue
        ref_mfcc = np.load(mfcc_file)
        distances[user] = compare_mfcc(test_mfcc, ref_mfcc)
        print(f"Distance to {user}: {distances[user]:.2f}")

    if not distances:
        print("No users found.")
        return

    closest_user, min_distance = min(distances.items(), key=lambda x: x[1])
    print(f"Closest match: {closest_user} (Distance: {min_distance:.2f})")

    if min_distance < THRESHOLD:
        print(f"Speaker identified as: {closest_user}")
    else:
        print("Speaker not recognized.")

def main():
    options = {
        "1": ("Register new user", register_user),
        "2": ("Identify speaker", identify_user),
        "3": ("Exit", None),
    }

    while True:
        print("\nVoice Biometric System")
        for key, (desc, _) in options.items():
            print(f"{key}. {desc}")

        choice = input("Choose: ").strip()
        if choice not in options:
            print("Invalid choice. Try again.")
            continue

        if choice == "3":
            break

        action = options[choice][1]
        if choice == "1":
            name = input("Enter user name: ").strip()
            if name:
                action(name)
            else:
                print("Name cannot be empty.")
        else:
            action()

if __name__ == "__main__":
    main()
