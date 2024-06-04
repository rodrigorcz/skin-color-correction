import cv2 as cv
import numpy as np
import mediapipe as mp # python 3.8 | 3.11
import scipy as sp
import pandas as pd
import math
from PIL import Image, ImageDraw

roi_input2 = roi_input.copy()

roi_input2 = cv.cvtColor(roi_input2,cv.COLOR_BGR2HSV)

roi_input2 = roi_input2.astype(np.float32)
mask = (roi_input2 == 0).all(axis=2)
roi_input2[mask] = np.nan


R =np.clip(np.nanmean(roi_input2[...,0]),0,255).astype(int)
g= np.clip(np.nanmean(roi_input2[...,1]),0,255).astype(int)
b =np.clip(np.nanmean(roi_input2[...,2]),0,255).astype(int)

Rs =np.clip(np.nanstd(roi_input2[...,0]),0,255).astype(int)
gs= np.clip(np.nanstd(roi_input2[...,1]),0,255).astype(int)
bs =np.clip(np.nanstd(roi_input2[...,2]),0,255).astype(int)

lower = (R-Rs, g - gs, b - bs)
upper = (R + Rs, g + gs, b + bs)
print(lower,upper)
roi_input2 = np.array(roi_input2)
mask2 = cv.inRange(cv.cvtColor(img_input,cv.COLOR_BGR2HSV),(-11, 94, 17), (27, 172, 100))
