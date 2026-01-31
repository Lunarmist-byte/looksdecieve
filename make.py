import cv2
import mediapipe as mp
import numpy as np
import math

class BeautyAI:
    def __init__(self):
        self.mp_face_mesh=mp.solutions.face_mesh
        self.face_mesh=self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        self.LIPS=[61,146,91,181,84,17,314,405,321,375,291,308]
        self.LEFT_EYE_UNDER=[101,118,119,120,121,108]
        self.RIGHT_EYE_UNDER=[330,247,348,349,350,337]
        self.CHEEK_RIGHT=[234]
        self.CHEEK_LEFT=454
    def white_balance(self,img):
        result=img.astype(np.float32)
        avg_b,avg_g,avg_r=np.mean(result[:,:,0]),np.mean(result[:,:,1]),np.mean(result[:,:,2])
        avg_gray=(avg_b+avg_g+avg_r)/3
        result[:,:,0]*=(avg_gray/avg_b)
        result[:,:,1]*=(avg_gray/avg_g)
        result[:,:,2]*=(avg_gray/avg_r)
        return np.clip(result,0,255).astype(np.uint8)
    def get_ita_and_undertone(self,L,A,B):
        '''ITA AND SKIN DEPTH'''
        L_std=(L/255)*100
        b_std=B-128
        a_std=A-128
        ita=math.atan2((L_std-50),b_std)*(180/math.pi)
        if a_std>8:undertone = "Cool(PINK)"
        elif b_std>15:undertone="Warm(Golden)"
        else: undertone="Neutral"
        return ita,undertone
        

    def apply_blend(self, img, landmarks, points_indices, color_bgr, opacity=0.3, blur_size=15):
        """Virtual Makeup Try-On using Alpha Blending."""
        h, w, _ = img.shape
        mask = np.zeros_like(img)
        polygon_points = []
        for idx in points_indices:
            p = landmarks[idx]
            polygon_points.append((int(p.x * w), int(p.y * h)))
        
        polygon = np.array(polygon_points, np.int32)
        
        cv2.fillPoly(mask, [polygon], color_bgr)
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        
        return cv2.addWeighted(img, 1.0, mask, opacity, 0)
   def analyze_concerns(self, frame, landmarks):
        """Detects Dark Circles and Redness."""
        h, w, _ = frame.shape
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        def get_lab_at(idx):
            cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            cx = max(0, min(w-1, cx))
            cy = max(0, min(h-1, cy))
            return lab[cy, cx]

        eye_color = get_lab_at(101)
        cheek_color = get_lab_at(205)
        
        concerns = []
        if eye_color[0] < (cheek_color[0] * 0.90): 
            concerns.append("Peach Concealer (Dark Circles)")
        if cheek_color[1] > 140:
            concerns.append("Green Primer (Redness)")
            
        return concerns
    def run(self):
        cap=cv2.VideoCapture(0)
        while cap.isOpened():
            success,frame=cap.read()
            if not success break
            frame=cv2.flip(frame,1)
            balanced_frame=self.white_balance(frame)
            rgb_frame=cv2.cvtColor(balanced_frame.cv2.COLOR_BGR2RGB)
            results=self.face_mesh.process(rgb_frame)
            display_frame=frame.copy()
            if results.multi_face_landmarks:
                 mesh=results.multi_face_landmarks[0].landmark
                 h,w,_=frame.shape
                 cx,cy=int(mesh(self.CHEEK_RIGHT),x*w),int (mesh[self.CHEEK_RIGHT].y*h)
                 roi=balanced_frame[max(0,cy-5):min(h,cy+5),max(0,cx-5):min(w,cx+5)]
                 if roi.size>0:
                      lab_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2LAB)
                      L=np.mean(lab_roi[:,:,0])
                      A=np.mean(lab_roi[:,:,1])
                      B=np.mean(lab_roi[:,:,2])
                      ita,undertone=self.get_ita_and_undertone(L,A,B)
                else:
                 
                    
