import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

#=============================================================================
#Remove duplicates for first part
#“I find that the harder I work, the more (bad) luck I seem to have.” — Thomas Jefferson
#=============================================================================

def remove_duplicate_frames(input_folder, output_folder, similarity_threshold=0.80): 
    # if output doesnt exist then make it 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frames = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')])

    # processed unique frames
    unique_frames = []  
    unique_count = 0  

    # iterate through frames
    for frame_name in frames:
        frame_path = os.path.join(input_folder, frame_name)
        current_frame = cv2.imread(frame_path)

        if current_frame is None:
            continue  

        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_current = cv2.GaussianBlur(gray_current, (5, 5), 0)  
        gray_current = cv2.normalize(gray_current, None, 0, 255, cv2.NORM_MINMAX) 

        # resizing
        resized_current = cv2.resize(gray_current, (64, 64)) 

        is_duplicate = False
        for unique_frame in unique_frames:
            # Compute the Structural Similarity Index (SSIM) between frames
            ssim_score, _ = ssim(resized_current, unique_frame, full=True)
            if ssim_score >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_count += 1
            print('new frame')
            unique_frames.append(resized_current)  

            output_frame_path = os.path.join(output_folder, f'frame_{unique_count:04d}.png')
            cv2.imwrite(output_frame_path, current_frame)

    print(f'Total unique frames saved: {unique_count}')
    return unique_count