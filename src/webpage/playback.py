from flask import Flask, request, render_template, jsonify, send_from_directory
import cv2
import os
import numpy as np
import shutil
from Remove_Duplicates import remove_duplicate_frames
from Remove_Duplicates_Highlight import remove_duplicate_frames_highlight
from align import align
from highlight import highlight_differences
import matlab.engine
from align import align  

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
FRAME_FOLDER = 'frames'
CROPPED_FRAME_FOLDER = 'cropped_frames'
UNIQUE_FRAME_FOLDER = 'unique_frames'
HIGHLIGHTED_OUTPUT = 'highlighted_output'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

current_video = None
current_video_one = None
current_video_two = None
points = []
points_one = []
points_two = []

#=============================================================================
#functions (used by many things later)
#That's one small step for man, one giant leap for mankind - Neil Armstrong
#=============================================================================
def cleanup_directories():
    if os.path.exists(UPLOAD_FOLDER):
        for file in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, file)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    for folder in [FRAME_FOLDER, CROPPED_FRAME_FOLDER, UNIQUE_FRAME_FOLDER, HIGHLIGHTED_OUTPUT]:
        if os.path.exists(folder):
            shutil.rmtree(folder)

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)] 
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered
#=============================================================================
#the different web pages
#carpe diem! - Horace (but im not doing a very good job at it) 
#=============================================================================
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/reconstruct')
def reconstruct():
    return render_template('reconstruct.html')

@app.route('/contact_us')
def contact():
    return render_template('contact_us.html')

@app.route('/compare')
def compare():
    return render_template('compare.html')

@app.route('/about_us')
def about():
    return render_template('about_us.html')

@app.route('/data')
def data():
   return render_template('data.html')
#=============================================================================
#for first button
#if it ain't broke dont fix it
#=============================================================================
@app.route('/upload', methods=['POST'])
def upload_video():
    global current_video
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        current_video = file_path

        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return jsonify({'error': 'Failed to extract the first frame'}), 500

        first_frame_path = os.path.join(UPLOAD_FOLDER, 'first_frame.jpg')
        cv2.imwrite(first_frame_path, frame)
        return jsonify({'frame_path': f'/uploads/first_frame.jpg', 'video_path': file_path}), 200
    return jsonify({'error': 'File upload failed'}), 500


@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/process', methods=['POST'])
def process_video():
    global current_video, points
    data = request.json
    if 'points' not in data:
        return jsonify({'error': 'Points not provided'}), 400

    points = [(p['x'], p['y']) for p in data['points']]

    if len(points) != 4:
        return jsonify({'error': 'Four points are required for processing'}), 400

    cap = cv2.VideoCapture(current_video)
    base_name = os.path.splitext(os.path.basename(current_video))[0]

    cropped_folder_path = os.path.join(CROPPED_FRAME_FOLDER, base_name)
    unique_folder_path = os.path.join(UNIQUE_FRAME_FOLDER, base_name)
    os.makedirs(CROPPED_FRAME_FOLDER, exist_ok=True)
    os.makedirs(UNIQUE_FRAME_FOLDER, exist_ok=True)

    for folder in [cropped_folder_path, unique_folder_path]:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                os.remove(os.path.join(folder, file))
        else:
            os.makedirs(folder)

    ret, frame = cap.read()
    if not ret:
        return jsonify({'error': 'Failed to load the video'}), 500

    ordered_points = order_points(points)

    output_width = int(np.linalg.norm(ordered_points[1] - ordered_points[0]))
    output_height = int(np.linalg.norm(ordered_points[2] - ordered_points[1]))
    dest_points = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ], dtype="float32")

    initial_points = np.array(ordered_points, dtype="float32").reshape(-1, 1, 2)
    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, gray_frame, initial_points, None)

        if status is not None and status.sum() == 4:
            current_points = order_points(current_points.reshape(-1, 2))
            homography_matrix = cv2.getPerspectiveTransform(current_points, dest_points)
            cropped_frame = cv2.warpPerspective(frame, homography_matrix, (output_width, output_height))

            cropped_filename = os.path.join(cropped_folder_path, f"cropped_{frame_count:04d}.png")
            os.makedirs(cropped_folder_path, exist_ok=True)
            cv2.imwrite(cropped_filename, cropped_frame)
            frame_count += 1
        else:
            break

        prev_frame = gray_frame.copy()
        initial_points = np.array(current_points, dtype="float32").reshape(-1, 1, 2)

    cap.release()

    unique_count = remove_duplicate_frames(
        cropped_folder_path,
        unique_folder_path,
        similarity_threshold=0.91
    )

    project_path = unique_folder_path
    try:

        eng = matlab.engine.start_matlab()

        eng.addpath(project_path)

        eng.construct3DModel(unique_folder_path, nargout=0)
        eng.quit()

        cleanup_directories()

        return jsonify({
            'message': '3D volume successfully constructed and visualized in MATLAB',
            'unique_count': unique_count
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500 
#=============================================================================
#uploads for second button
#“Less is more.” — Mies van der Rohe. (;-;)
#============================================================================= 

@app.route('/uploadone', methods=['POST'])
def upload_video_one():
    global current_video_one
    if 'file' not in request.files:
        return jsonify({'error': 'No File Was Selected'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No File Was Selected'}), 400
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, 'video_one.mp4')
        file.save(file_path)
        current_video_one = file_path
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return jsonify({'error': 'Failed to extract the first frame for video one'}), 500
        first_frame_path = os.path.join(UPLOAD_FOLDER, 'video_one_frame.jpg')
        cv2.imwrite(first_frame_path, frame)
        return jsonify({'frame_path': f'/uploads/video_one_frame.jpg'}), 200
    return jsonify({'error': 'File upload failed for video one'}), 500

@app.route('/uploadtwo', methods=['POST'])
def upload_video_two():
    global current_video_two
    if 'file' not in request.files:
        return jsonify({'error': 'No File Was Selected'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No File Was Selected'}), 400
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, 'video_two.mp4')
        file.save(file_path)
        current_video_two = file_path
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return jsonify({'error': 'Failed to extract the first frame for video two'}), 500

        first_frame_path = os.path.join(UPLOAD_FOLDER, 'video_two_frame.jpg')
        cv2.imwrite(first_frame_path, frame)
        return jsonify({'frame_path': f'/uploads/video_two_frame.jpg'}), 200
    return jsonify({'error': 'File upload failed for video two'}), 500

#=============================================================================
#for second button process
#“I think (it will work), therefore I am.” — René Descartes.
#============================================================================= 
@app.route('/processone', methods=['POST'])
def process_one():
    global current_video_one
    if not current_video_one:
        return jsonify({'error': 'Video one not uploaded'}), 400

    data = request.json
    if 'points' not in data:
        return jsonify({'error': 'Points not provided for video one'}), 400
    points_one = [(p['x'], p['y']) for p in data['points']]
    if len(points_one) != 4:
        return jsonify({'error': 'Four points are required for video one'}), 400

    return process_video_logic(current_video_one, "video_one", points_one)

#processes run seperately at first, then all together later

@app.route('/processtwo', methods=['POST'])
def process_two():
    global current_video_two
    if not current_video_two:
        return jsonify({'error': 'Video two not uploaded'}), 400

    data = request.json
    if 'points' not in data:
        return jsonify({'error': 'Points not provided for video two'}), 400
    points_two = [(p['x'], p['y']) for p in data['points']]
    if len(points_two) != 4:
        return jsonify({'error': 'Four points are required for video two'}), 400

    return process_video_logic(current_video_two, "video_two", points_two)

#=============================================================================
#for second button process (logic)
#something i dont have right now
#============================================================================= 
def process_video_logic(video_path, video_name, points):
    cap = cv2.VideoCapture(video_path)
    cropped_folder_path = os.path.join(CROPPED_FRAME_FOLDER, video_name)
    unique_folder_path = os.path.join(UNIQUE_FRAME_FOLDER, video_name)
    os.makedirs(cropped_folder_path, exist_ok=True)
    os.makedirs(unique_folder_path, exist_ok=True)

    ret, frame = cap.read()
    if not ret:
        return jsonify({'error': f'Failed to load the video: {video_name}'}), 500

    ordered_points = order_points(points)

    output_width = int(np.linalg.norm(ordered_points[1] - ordered_points[0]))
    output_height = int(np.linalg.norm(ordered_points[2] - ordered_points[1]))
    dest_points = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ], dtype="float32")

    initial_points = np.array(ordered_points, dtype="float32").reshape(-1, 1, 2)
    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, gray_frame, initial_points, None)

        if status is not None and status.sum() == 4:
            current_points = order_points(current_points.reshape(-1, 2))
            homography_matrix = cv2.getPerspectiveTransform(current_points, dest_points)
            cropped_frame = cv2.warpPerspective(frame, homography_matrix, (output_width, output_height))

            cropped_filename = os.path.join(cropped_folder_path, f"cropped_{frame_count:04d}.png")
            cv2.imwrite(cropped_filename, cropped_frame)
            frame_count += 1
        else:
            break

        prev_frame = gray_frame.copy()
        initial_points = np.array(current_points, dtype="float32").reshape(-1, 1, 2)

    cap.release()

    unique_count = remove_duplicate_frames_highlight(
        cropped_folder_path,
        unique_folder_path,
        similarity_threshold=0.99
    )

    return jsonify({
        'message': f'{video_name} processed successfully',
        'unique_count': unique_count
    }), 200

#=============================================================================
#for second button modelling and highlight and stuff
#You miss 100% of the shots you don’t take. (but i took the shots and missed all of them anyway)
#============================================================================= 
@app.route('/constructmodel', methods=['POST'])
def construct_model():
    unique_folder_one = os.path.join(UNIQUE_FRAME_FOLDER, "video_one")
    unique_folder_two = os.path.join(UNIQUE_FRAME_FOLDER, "video_two")
    
    aligned_folder = "aligned_frames"
    aligned_folder_one = os.path.join(aligned_folder, "video_one")
    aligned_folder_two = os.path.join(aligned_folder, "video_two")
    
    highlighted_folder = "highlighted_output"

    try:
        if not os.path.exists(unique_folder_one) or not os.path.exists(unique_folder_two):
            return jsonify({'error': 'One or both unique frame folders are missing. Make sure /processone and /processtwo are run first.'}), 400

        align(unique_folder_one, unique_folder_two, aligned_folder)
        print(f"Alignment completed. Aligned frames saved in {aligned_folder_one} and {aligned_folder_two}.")

        highlight_differences(aligned_folder_one, aligned_folder_two, highlighted_folder, noise_threshold=60)
        print(f"Differences highlighted. Results saved in {highlighted_folder}.")

        try:

            eng = matlab.engine.start_matlab()

            eng.addpath(highlighted_folder)

            eng.construct3DModelhighlight()
            eng.quit()
            print("3D model successfully constructed and visualized in MATLAB.")

            cleanup_directories()


            return jsonify({
                'message': 'Alignment, highlighting, and 3D modeling completed successfully and visualized in MATLAB.',
                'aligned_frames': {
                    'video_one': aligned_folder_one,
                    'video_two': aligned_folder_two
                },
                'highlighted_frames': highlighted_folder
            }), 200

        except Exception as matlab_error:
            print(f"MATLAB error: {matlab_error}")
            return jsonify({'error': f'MATLAB visualization failed: {str(matlab_error)}'}), 500

    except Exception as e:
        return jsonify({'error': f'An error occurred during processing: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)