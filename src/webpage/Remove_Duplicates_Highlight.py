import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

#=============================================================================
#Remove duplicates for Highlighting
#“Never regret anything that made you smile. (what about choices that made me cry)”  — Mark Twain
#=============================================================================

def remove_duplicate_frames_highlight(input_folder, output_folder, similarity_threshold=0.99):
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

        red_channel = current_frame[:, :, 2]

        other_channels = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        highlighted_frame = other_channels.copy()
        highlighted_frame[red_channel > 50] = red_channel[red_channel > 50]

        normalized_highlighted = cv2.normalize(highlighted_frame, None, 0, 255, cv2.NORM_MINMAX)

        # resizing
        resized_frame = cv2.resize(normalized_highlighted, (64, 64))

        is_duplicate = False

        for unique_frame in unique_frames:
            # Compute the Structural Similarity Index (SSIM) between frames
            ssim_score, _ = ssim(resized_frame, unique_frame, full=True)
            if ssim_score >= similarity_threshold:  # Check if the similarity is above the threshold
                is_duplicate = True
                break

        # If the frame unique 
        if not is_duplicate:
            unique_count += 1  
            unique_frames.append(resized_frame)  

            output_frame_path = os.path.join(output_folder, f'frame_{unique_count:04d}.png')
            cv2.imwrite(output_frame_path, current_frame)

    print(f'Total unique frames saved: {unique_count}')
    return unique_count
