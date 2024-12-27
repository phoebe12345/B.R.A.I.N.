function volume3D = construct3DModel(folderPath)
    %=============================================================================
    %highlighting with color (button 2 for highlighting)
    %“Life is either a daring adventure or nothing.” — Helen Keller
    %=============================================================================

    downsampleFactor = 0.5;  
    frameInterval = 1;       
    blankSlices = 1;         

    frameFiles = dir(fullfile(folderPath, '*.png'));
    disp(['Number of .png files found: ', num2str(numel(frameFiles))]);

    if isempty(frameFiles)
        error('No PNG files found in the specified folder.');
    end

    firstFrame = imread(fullfile(folderPath, frameFiles(1).name));
    firstFrameGray = rgb2gray(firstFrame);
    firstFrameGray = imresize(firstFrameGray, downsampleFactor);

    % Calculate the number of frames to process and total slices in the volume
    numFrames = floor(length(frameFiles) / frameInterval);
    newNumFrames = numFrames + (numFrames - 1) * blankSlices;

    volume3D = zeros([size(firstFrameGray), newNumFrames], 'uint8');

    frameIdx = 1;
    volumeIdx = 1;

    for i = 1:frameInterval:length(frameFiles)
        frame = imread(fullfile(folderPath, frameFiles(i).name));
        grayFrame = rgb2gray(frame);
        grayFrame = imresize(grayFrame, downsampleFactor);

        mask = imbinarize(grayFrame, 'adaptive', 'Sensitivity', 0.5);
        mask = imclose(mask, strel('disk', 5)); 
        mask = imfill(mask, 'holes'); 

        objectOnly = grayFrame;
        objectOnly(~mask) = 0;

        volume3D(:, :, volumeIdx) = objectOnly;

        volumeIdx = volumeIdx + 1 + blankSlices;
    end

    % Display the volume dimensions
    disp(['3D volume shape with spacing: ', num2str(size(volume3D))]);

    % Visualize the 3D volume
    if exist('volshow', 'file')
        fig = uifigure('Name', '3D Volume Viewer', 'Position', [100 100 800 600]);
        volshow(volume3D, 'Parent', fig);
        uiwait(fig); 
    else
        figure;
        for i = 1:size(volume3D, 3)
            imshow(volume3D(:, :, i), []);
            title(['Slice ', num2str(i)]);
            pause(0.1);
        end
    end
end