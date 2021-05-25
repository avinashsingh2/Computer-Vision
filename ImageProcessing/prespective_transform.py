import cv2
import numpy as np

class Transform:
    def __init__(self, frame_width, frame_height):
        self.width = frame_width
        self.height = frame_height
        self.out_pts = np.float32([[0,0],[self.width,0],[0,self.height],[self.width,self.height]])
    def prespective_transform(self, image, pts):
        #expecting points as [[111,219],[287,188],[154,482],[352,440]]
        pts = np.float32(pts)
        transform_mat = cv2.getPerspectiveTransform(pts, self.out_pts)
        transform_img = cv2.warpPerspective(image,transform_mat,(self.width,self.height))
        return transform_img


