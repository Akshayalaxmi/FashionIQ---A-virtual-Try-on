
from pose_estimator import find_rotation_angle
import cv2 as cv
import utils
import numpy as np


def get_shoulder_loc_mannual(cloth_seg):

    start_crop, crop_seg = utils.crop_square(cloth_seg)
    crop_seg = cv.cvtColor(crop_seg, cv.COLOR_BGR2GRAY)
    height, width = crop_seg.shape[:2]
    offset = int(0.15*width)
    col_vector = crop_seg[:, offset]
    right_shoulder = (offset+start_crop[1], min(np.where(col_vector!=0)[0])+start_crop[0])
    col_vector = crop_seg[:, width-offset]
    left_shoulder = (width-offset+start_crop[1], min(np.where(col_vector!=0)[0]+start_crop[0]))

    shoulder_points = [right_shoulder, left_shoulder]
    print("Shoulder Points:",shoulder_points)

    return shoulder_points

def get_shoulder_details_mannual(cloth_seg):
    try:
        shoulder_points = get_shoulder_loc_mannual(cloth_seg)

        distance = shoulder_points[1][0] - shoulder_points[0][0]
        return shoulder_points, distance
    except Exception as e:
        print("Error at manual shoulder detection.\n"+str(e))
        raise Exception("Invalid Source Image.")
