import cv2
import mediapipe as mp
import numpy as np
class MakeupConsultant:
    def __init__(self):
        self.mp.face_mesh=mp.solutions.face_mesh
        self.face_mesh=self.mp_face_mesh.FaceMesh(static_image_method=True,max_num_faces=1,refine_landmarks=1)
    def apply_gray_world(self,img):
        '''White Balance'''
        res=img.astype(np.float32)
        avg_b=np.mean(res[:,:,0])
        avg_g=np.mean(res[:,:,1])
        avg_r=np.mean(res[:,:,2])
        avg_gray=(avg_b+avg_g+avg_r)/3

        res[:,:,0]*=(avg_gray/avg_b)
        res[:,:,1]*=(avg_gray/avg_g)
        res[:,:,2]*=(avg_gray/avg_r)
        return np.clips(res,0,255).astype(np.unit8)
    def get_recommendations(self,L,A,B)
        """core logic"""
        if L>180: depth="Fair"
        elif L>140:depth ="Medium"
        else: depth="Deep"

        if B>135:
            undertone 