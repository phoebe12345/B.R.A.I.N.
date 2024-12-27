import cv2
import os

#=============================================================================
#highlighting function 
#“Don’t stop believin’.” — Journey, “Don’t Stop Believin’” 
#=============================================================================

def highlight_differences(folder1, folder2, output_folder='highlighted_output', noise_threshold = 30):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frames1 = sorted([f for f in os.listdir(folder1) if f.endswith(('png', 'jpg', 'jpeg'))])
    frames2 = sorted([f for f in os.listdir(folder2) if f.endswith(('png', 'jpg', 'jpeg'))])
    
    if len(frames1) != len(frames2):
        print("Error: The two folders must have the same number of frames.")
        return
    
    for i, (frame1_name, frame2_name) in enumerate(zip(frames1, frames2)):
        frame1_path = os.path.join(folder1, frame1_name)
        frame2_path = os.path.join(folder2, frame2_name)
        
        frame1 = cv2.imread(frame1_path)
        frame2 = cv2.imread(frame2_path)
        
        if frame1 is None or frame2 is None:
            print(f"Error: Could not load frames {frame1_name} and {frame2_name}.")
            continue
        
        diff = cv2.absdiff(frame1, frame2)
        
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh_diff = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        
        highlighted_frame = frame1.copy()
        highlighted_frame[thresh_diff > 0] = [0, 0, 255]  
        
        output_path = os.path.join(output_folder, frame1_name)
        cv2.imwrite(output_path, highlighted_frame)
        
        print(f"Processed frame {i+1}/{len(frames1)}: {frame1_name}")
    
    print(f"All frames processed. Highlighted frames are saved in '{output_folder}'.")