import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_histogram(image):
    hist, _ = np.histogram(image.flatten(), 256, [0,256])
    pdf = hist / hist.sum()
    cdf = pdf.cumsum()
    cdf_normalized = cdf * hist.max()
    return pdf, cdf, cdf_normalized

def histogram_matching(source, template):
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    
    _, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    
    return interp_t_values[bin_idx].reshape(oldshape)

def save_histogram_and_cdf(image, label, output_dir):
    pdf, cdf, _ = calculate_histogram(image)
    
    plt.figure()
    plt.plot(pdf)
    plt.savefig(os.path.join(output_dir, f'pdf_{label.lower()}.png'))
    plt.close()

    plt.figure()
    plt.plot(cdf)
    plt.savefig(os.path.join(output_dir, f'cdf_{label.lower()}.png'))
    plt.close()

def save_results(input_image, target_image, matched_image, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    

    cv2.imwrite(os.path.join(output_dir, 'input_image.png'), input_image)
    cv2.imwrite(os.path.join(output_dir, 'target_image.png'), target_image)
    cv2.imwrite(os.path.join(output_dir, 'matched_image.png'), matched_image)
    
 
    save_histogram_and_cdf(input_image, 'Entrada', output_dir)
    save_histogram_and_cdf(target_image, 'Alvo', output_dir)
    save_histogram_and_cdf(matched_image, 'Ajustada', output_dir)

def main(input_image_path, target_image_path, output_dir):

    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    
    matched_image = histogram_matching(input_image, target_image)
    
    save_results(input_image, target_image, matched_image, output_dir)

input_image_path = 'img/e2.jpg'  
target_image_path = 'img/e33.jpg'  
output_dir = 'paper/'  
main(input_image_path, target_image_path, output_dir)