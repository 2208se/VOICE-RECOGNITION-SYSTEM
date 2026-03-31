import os
from record import record_voice
from features import extract_and_save_mfcc
from compare import compare_mfcc
import numpy as np

USERS_FOLDER = "users"

def register_user(name):
    user_path = os.path.join(USERS_FOLDER, name)
    wav_path = os.path.join(user_path, "voice.wav")
    mfcc_path = os.path.join(user_path, "mfcc.npy")

    print(f"Registering user: {name}")
    record_voice(wav_path)
    extract_and_save_mfcc(wav_path, mfcc_path)
    print(f"✅ {name} registered.")

def identify_user():
    temp_path = "temp.wav"
    record_voice(temp_path)
    test_mfcc = extract_and_save_mfcc(temp_path, "temp.npy")

    distances = {}
    for user in os.listdir(USERS_FOLDER):
        mfcc_path = os.path.join(USERS_FOLDER, user, "mfcc.npy")
        if not os.path.exists(mfcc_path):
            continue
        ref_mfcc = np.load(mfcc_path)
        dist = compare_mfcc(test_mfcc, ref_mfcc)
        distances[user] = dist
        print(f"📏 Distance to {user}: {dist:.2f}")

    if not distances:
        print("❌ No users found.")
        return

    min_user = min(distances, key=distances.get)
    min_distance = distances[min_user]
    threshold = 15000  # You can tune this number based on real testing

    print(f"\n🔍 Closest match: {min_user} (Distance: {min_distance:.2f})")

    if min_distance < threshold:
        print(f"✅ Speaker identified as: {min_user}")
    else:
        print("❌ Speaker not recognized.")

def main():
    while True:
        print("\n🔐 Voice Biometric System")
        print("1. Register new user")
        print("2. Identify speaker")
        print("3. Exit")
        choice = input("Choose: ")

        if choice == "1":
            name = input("Enter user name: ")
            register_user(name)
        elif choice == "2":
            identify_user()
        elif choice == "3":
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
