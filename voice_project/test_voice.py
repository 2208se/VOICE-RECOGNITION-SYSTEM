import os
from record import record_voice
from features import extract_and_save_mfcc
from compare import compare_mfcc
import numpy as np

USERS_FOLDER = "users"
THRESHOLD = 15000 

def load_registered_users():
    users = {}
    for user in os.listdir(USERS_FOLDER):
        mfcc_path = os.path.join(USERS_FOLDER, user, "mfcc.npy")
        if os.path.exists(mfcc_path):
            users[user] = np.load(mfcc_path)
    return users

def identify(test_mfcc, registered_users):
    distances = {user: compare_mfcc(test_mfcc, mfcc) 
                 for user, mfcc in registered_users.items()}
    if not distances:
        return None, None
    closest_user, min_distance = min(distances.items(), key=lambda x: x[1])
    if min_distance < THRESHOLD:
        return closest_user, min_distance
    return None, min_distance

def test_performance(trials_per_user=2, duration=3):
    registered_users = load_registered_users()
    if not registered_users:
        print("No registered users found. Register some first.")
        return

    total_tests = 0
    correct = 0

    for user in registered_users:
        print(f"\nTesting user: {user}")
        for i in range(trials_per_user):
            total_tests += 1
            print(f" Trial {i+1}: Record your voice for testing.")
            temp_wav = f"temp_test_{user}_{i}.wav"
            temp_mfcc = f"temp_test_{user}_{i}.npy"
            
            record_voice(temp_wav, duration=duration)
            test_mfcc = extract_and_save_mfcc(temp_wav, temp_mfcc)
            
            identified_user, distance = identify(test_mfcc, registered_users)
            if identified_user == user:
                print(f"  Correctly identified ({distance:.2f})")
                correct += 1
            else:
                print(f"  Incorrectly identified as {identified_user} ({distance:.2f})")

            # Clean up temporary files
            os.remove(temp_wav)
            os.remove(temp_mfcc)

    accuracy = (correct / total_tests) * 100
    print(f"\nPerformance test completed. Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    test_performance(trials_per_user=2, duration=3)
