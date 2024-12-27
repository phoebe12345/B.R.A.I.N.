import cv2
import os
import numpy as np
#=============================================================================
#alinging
#it wasnt me
#=============================================================================

def align(volume1_folder, volume2_folder, output_folder):
    aligned_folder1 = os.path.join(output_folder, "video_one")
    aligned_folder2 = os.path.join(output_folder, "video_two")

    # CLEAN THEM FOLDERS 
    for folder in [aligned_folder1, aligned_folder2]:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                os.remove(os.path.join(folder, file))
        else:
            os.makedirs(folder)

    def resize_and_pad(image, target_size):

        target_width, target_height = target_size
        h, w = image.shape[:2]
        scale = min(target_width / w, target_height / h)

        # resize so modelling and highlighting doesnt complain
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # adding 'spce' if its too small (colors AND no colors)
        if len(image.shape) == 2:  
            padded = np.zeros((target_height, target_width), dtype=image.dtype)
        else:  
            padded = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)

        #offset so we dont break the modeeling
        x_offset = (target_width - new_w) // 2
        y_offset = (target_height - new_h) // 2

        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return padded

    def align_image(reference_img, to_align_img, target_size):
        # SIFT
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(reference_img, None)
        keypoints2, descriptors2 = sift.detectAndCompute(to_align_img, None)

        # FLANN
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        # lowes ratio test 
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 10: 
            print("Insufficient matches. Falling back to resize and pad.")
            return resize_and_pad(to_align_img, target_size)

        points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        matrix, _ = cv2.estimateAffinePartial2D(points2, points1, method=cv2.RANSAC)

        aligned_image = cv2.warpAffine(to_align_img, matrix, (reference_img.shape[1], reference_img.shape[0]))

        return resize_and_pad(aligned_image, target_size)

    frames1 = sorted([f for f in os.listdir(volume1_folder) if f.endswith(('png', 'jpg', 'jpeg'))])
    frames2 = sorted([f for f in os.listdir(volume2_folder) if f.endswith(('png', 'jpg', 'jpeg'))])

    if not frames1:
        raise ValueError(f"No images found in folder: {volume1_folder}")

    reference_image_path = os.path.join(volume1_folder, frames1[0])
    reference_image = cv2.imread(reference_image_path)

    if reference_image is None:
        raise ValueError(f"Failed to read the reference image: {reference_image_path}")

    target_size = (reference_image.shape[1], reference_image.shape[0])  
   
    for i, (frame1_name, frame2_name) in enumerate(zip(frames1, frames2)):
        frame1_path = os.path.join(volume1_folder, frame1_name)
        frame2_path = os.path.join(volume2_folder, frame2_name)

        frame1 = cv2.imread(frame1_path)
        frame2 = cv2.imread(frame2_path)

        if frame1 is None or frame2 is None:
            print(f"Error reading images: {frame1_name}, {frame2_name}. Skipping...")
            continue

        aligned_frame2 = align_image(frame1, frame2, target_size)

        aligned_frame1 = resize_and_pad(frame1, target_size)

        output_file1 = os.path.join(aligned_folder1, frame1_name)
        output_file2 = os.path.join(aligned_folder2, frame2_name)
        cv2.imwrite(output_file1, aligned_frame1) 
        cv2.imwrite(output_file2, aligned_frame2)  

        print(f"Processed frame {i + 1}/{len(frames1)}: {frame1_name}")

    print(f"Alignment complete. Frames saved in '{output_folder}'.")
