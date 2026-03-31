import os
import numpy as np
from record import record_voice
from features import extract_and_save_mfcc

def test_full_pipeline():
    test_folder = "tests/temp_test"
    os.makedirs(test_folder, exist_ok=True)

    wav_path = os.path.join(test_folder, "test.wav")
    mfcc_path = os.path.join(test_folder, "test_mfcc.npy")


    record_voice(wav_path, duration=2)

    # Extract MFCC features
    mfcc = extract_and_save_mfcc(wav_path, mfcc_path)

    assert os.path.exists(wav_path), "WAV file was not created."
    assert os.path.exists(mfcc_path), "MFCC file was not saved."
    assert isinstance(mfcc, np.ndarray), "MFCC is not a NumPy array."
    assert mfcc.size > 0, "MFCC array is empty."


    os.remove(wav_path)
    os.remove(mfcc_path)
    os.rmdir(test_folder)

if __name__ == "__main__":
    test_full_pipeline()
    print("Integration test passed.")
