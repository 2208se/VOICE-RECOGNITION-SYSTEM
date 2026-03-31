
from fastdtw import fastdtw
from scipy.spatial.distance import cityblock  # Manhattan distance

def compare_mfcc(mfcc1, mfcc2):
    distance, _ = fastdtw(mfcc1, mfcc2, dist=cityblock)
    return distance

