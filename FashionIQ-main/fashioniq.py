from pose_estimator import PoseEstimator
import custom_shoulder_locator
from segmentation import cloth_extractor
import utils

import cv2 as cv
import imutils
import logging


ERROR_LIST = [] 
MIN_SHOULDER_DISTANCE=0

class Vton:
    def __init__(self, source_img, dest_img):
        self.source_img = source_img
        # print(self.source_img)
        self.dest_img = dest_img
        # print(cv.imread(self.dest_img))
        self.dest_pose_estimator = PoseEstimator(cv.imread(self.dest_img))
        self.source_pose_estimator = None  
        self.error_list = []

    def cloth_segmentation(self):
        try:
            source_img, source_seg = cloth_extractor.extract_cloth(self.source_img)
        
            source_img = cv.cvtColor(source_img, cv.COLOR_RGB2BGR)
            source_seg = cv.cvtColor(source_seg, cv.COLOR_RGB2BGR)
            source_seg = utils.fill_holes(source_img, source_seg)

            return source_img, source_seg
        except Exception:
            raise Exception("Source image without cloth.")

    def get_source_shoulder_details(self, source_img, cloth_seg):
        try:
            self.source_pose_estimator = PoseEstimator(source_img)
            source_distance = self.source_pose_estimator.get_shoulder_details()
            source_points = self.source_pose_estimator.get_shoulder_points()
        except Exception as e:
            print(str(e))
            self.error_list.append("Issue in source image:"+str(e))
            logging.warning("Using manual shoulder detection for source image.")
            source_points, source_distance = custom_shoulder_locator.get_shoulder_details_mannual(cloth_seg)

        return source_points, source_distance

    def apply_cloth(self):
        try:
            dest_distance = self.dest_pose_estimator.get_shoulder_details()
            # print(dest_distance)
        except Exception as e:
            raise Exception("Issue in profile image:"+str(e))
        source_img, source_seg = self.cloth_segmentation()

        source_points, source_distance = self.get_source_shoulder_details(
            source_img, source_seg)

        if dest_distance < MIN_SHOULDER_DISTANCE:
            raise Exception("Shoulder detection issue in profile image.")
        if source_distance < MIN_SHOULDER_DISTANCE:
            raise Exception("Shoulder detection issue in source image.")
        resize_factor = dest_distance/source_distance
        print("resize factor:", resize_factor)

        source_seg = cv.resize(source_seg,
                               (int(source_seg.shape[1]*resize_factor),
                                int(source_seg.shape[0]*resize_factor))
                               )

        source_points[0] = utils.resize_shoulder_coord(
            source_points[0], resize_factor)
        source_points[1] = utils.resize_shoulder_coord(
            source_points[1], resize_factor)
        
        _, source_seg = utils.remove_segmentation_border(source_seg)

        dest_frame = cv.imread(self.dest_img)
        dest_points = self.dest_pose_estimator.get_shoulder_points()
        try:
            final_img = utils.blend_images(
                source_seg, source_points, dest_frame, dest_points)
            # print(final_img)
        except AssertionError:
            print("Assertion Error in blending images.")
            raise Exception("Issue in blending Images.")
        except Exception as e:
            print(str(e))
            raise Exception("Issue in blending Images.")
        return final_img, self.error_list
