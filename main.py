import cv2 as cv
import numpy as np
import mediapipe as mp # python [3.8 ~ 3.11]
import scipy as sp
import pandas as pd
import math
from PIL import Image, ImageDraw

# regions of interest (roi)
forehead = [10,338,297,332,298,293,334,296,336,9,107,66,105,63,68,103,67,109,10]
left_cheek= [116,117,118,119,100,142,203,206,216,192,213,147,123,116]
right_cheek = [345,346,347,348,329,371,423,426,436,416,433,376,352,345]
face = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10]

DESIRED_HEIGHT = 600
DESIRED_WIDTH = 600

# function to display images
def resize_and_show(image):
    h, w = image.shape[:2]

    if h < w:
      img = cv.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
      img = cv.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))

    cv.imshow('output',img)
    cv.waitKey(0) 
    cv.destroyAllWindows()

# selection of the regions of interest (roi)
def region_of_interest_segmentation(roi, info, img):
    imArray = np.asarray(img)

    polygon = []
    for i in range(len(roi)):
        idx_s = roi[i]
        start_point = (info[idx_s][3], info[idx_s][4])
        polygon.append(start_point)

    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
    mask = np.array(maskIm)

    newImArray = np.zeros(img.shape, dtype='uint8')

    newImArray[:,:,0] = np.multiply(imArray[:,:,0], mask)
    newImArray[:,:,1] = np.multiply(imArray[:,:,1], mask)
    newImArray[:,:,2] = np.multiply(imArray[:,:,2], mask)

    return newImArray, mask

# function to segmentation an input image
def segmentation(image, type = 2):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces = 1, min_detection_confidence=0.5)

    results = face_mesh.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    landmarks = results.multi_face_landmarks[0]

    shape_a = image.shape[1]
    shape_b = image.shape[0]
    info = []

    for landmark in landmarks.landmark:
        x = landmark.x
        y = landmark.y
        z = landmark.z

        relative_x = int(shape_a * x)
        relative_y = int(shape_b * y)

        info.append([x,y,z,relative_x,relative_y])

    # roi = forehead + left_cheek + right_cheek
    if type == 1:
        img1,_ = region_of_interest_segmentation(forehead, info, image)
        img2,_ = region_of_interest_segmentation(left_cheek, info, image)
        img3,_ = region_of_interest_segmentation(right_cheek, info, image)
        regions_interest = np.maximum(np.maximum(img1, img2), img3)

    # roi = face
    if type == 2:
        img1,_ = region_of_interest_segmentation(face, info, image)
        regions_interest = img1

    return regions_interest

# color_grading - iterative distribution transfer (IDT) algorithm
def color_grading_idt(img1, img2, bins=300, n_rot=10, relaxation=1):

    n_dims = img1.shape[1]

    d1 = img1.T
    d2 = img2.T

    for i in range(n_rot):
        # generate a random orthogonal rotation matrix
        rot = sp.stats.special_ortho_group.rvs(n_dims).astype(np.float32)

        d1r = np.dot(rot, d1)
        d2r = np.dot(rot, d2)
        d_r = np.empty_like(d1)

        # perform the histogram matching for each dimension separately
        for j in range(n_dims):

            # filter out NaN values from the rotated datasets
            valid_d1r = d1r[j][~np.isnan(d1r[j])]
            valid_d2r = d2r[j][~np.isnan(d2r[j])]

            if len(valid_d1r) == 0 or len(valid_d2r) == 0:
                continue

            lo = min(valid_d1r.min(), valid_d2r.min())
            hi = max(valid_d1r.max(), valid_d2r.max())

            # compute histograms and cumulative distributions (CDF) for both datasets
            p1r, edges = np.histogram(valid_d1r, bins=bins, range=[lo, hi])
            p2r, _ = np.histogram(valid_d2r, bins=bins, range=[lo, hi])

            cp1r = p1r.cumsum().astype(np.float32)
            cp1r /= cp1r[-1]

            cp2r = p2r.cumsum().astype(np.float32)
            cp2r /= cp2r[-1]

            f = np.interp(cp1r, cp2r, edges[1:])

            d_r[j] = np.interp(d1r[j], edges[1:], f, left=lo, right=hi)

        d1 = relaxation * np.linalg.solve(rot, (d_r - d1r)) + d1

    return d1.T

# function to apply bilateral filter
def apply_bilateral(img):
    face_area = np.count_nonzero(img)  
    total_area = img.shape[0] * img.shape[1]
    face_ratio = face_area / total_area

    # dynamic parameters
    d = max(5, int(15 * face_ratio))  # windows size
    sigma_color = 50 + int(50 * face_ratio)  
    sigma_space = 50 + int(50 * face_ratio)  

    # bilateral filter
    filtered_img = cv.bilateralFilter(img, d, sigma_color, sigma_space)

    return filtered_img

def main():

    # google monk skin tone examples
    path_input = "img/in/mst_input/skin_tone_5/img5.jpg"
    path_target = "img/in/golden_pics_mst/skin_tone_5/img1.jpg"

    img_input = cv.imread(path_input)
    img_target = cv.imread(path_target)

    # Resizing the images for optimization
    width = int(img_input.shape[1] * 40 / 100)
    height = int(img_input.shape[0] * 40 / 100)
    dim = (width, height)

    img_input = cv.resize(img_input, dim)
    img_target = cv.resize(img_target, dim)

    # segmentation of the regions of interest (roi)
    roi_input = cv.cvtColor(segmentation(img_input,2), cv.COLOR_BGR2RGB)
    roi_target = cv.cvtColor(segmentation(img_target,2), cv.COLOR_BGR2RGB)

    # creation of Dataframes
    df_roi_ipt = pd.DataFrame(roi_input.reshape(-1, roi_input.shape[-1]), columns=['r', 'g', 'b'])
    df_roi_tgt = pd.DataFrame(roi_target.reshape(-1, roi_target.shape[-1]), columns=['r', 'g', 'b'])

    # turning all black pixels into NaN
    df_roi_ipt_nan = df_roi_ipt.mask((df_roi_ipt[['r', 'g', 'b']] == 0).all(axis=1))
    df_roi_tgt_nan = df_roi_tgt.mask((df_roi_tgt[['r', 'g', 'b']] == 0).all(axis=1))

    a_input = df_roi_ipt_nan.values
    a_target = df_roi_tgt_nan.values

    # applying color grading
    a_result = color_grading_idt(a_input, a_target, bins=300, n_rot=20, relaxation=1)
    
    # converting the result back to an image
    img_output = a_result.reshape(img_input.shape)
    img_output = cv.convertScaleAbs(img_output)
    img_output = cv.cvtColor(img_output, cv.COLOR_RGB2BGR)

    # final processing
    img_bilateral = apply_bilateral(img_output)
    img_final = cv.add(img_input, img_bilateral)

    cv.imwrite("img/out/output1.jpg", img_final)

# main program
if __name__ == '__main__':
    main()