function volume3D = construct3DModelhighlight()
    %=============================================================================
    %highlighting
    %The star of the show  
    %=============================================================================
 
    folderPath = 'highlighted_output';  
    downsampleFactor = 0.5;
    frameInterval = 1;
    blankSlices = 1;

    frameFiles = dir(fullfile(folderPath, '*.png'));
    numFrames = floor(length(frameFiles) / frameInterval);

    firstFrame = imread(fullfile(folderPath, frameFiles(1).name));
    firstFrameResized = imresize(firstFrame, downsampleFactor);
    [targetHeight, targetWidth, ~] = size(firstFrameResized); 

    newNumFrames = numFrames + (numFrames - 1) * blankSlices;

    % Initialize 3D volume for RGB image
    volume3D = zeros([targetHeight, targetWidth, 3, newNumFrames], 'uint8');

    frameIdx = 1;
    volumeIdx = 1;

    for i = 1:frameInterval:length(frameFiles)
        % Read / resize 
        frame = imread(fullfile(folderPath, frameFiles(i).name));
        resizedFrame = imresize(frame, downsampleFactor);
        [currentHeight, currentWidth, ~] = size(resizedFrame);

        % Check if the dims of frame match reference size
        if currentHeight ~= targetHeight || currentWidth ~= targetWidth
            % If not resize
            disp(['Adjusting frame ', frameFiles(i).name, ' to match the target dimensions.']);
            resizedFrame = imresize(resizedFrame, [targetHeight, targetWidth]);
        end

        %greyscale 
        grayFrame = rgb2gray(resizedFrame);

        mask = imbinarize(grayFrame, 'adaptive', 'Sensitivity', 0.5);
        mask = imclose(mask, strel('disk', 5));  
        mask = imfill(mask, 'holes');            

        % Athis is just for background stuff so it doesnt look soo closely at that 
        objectOnly = resizedFrame; 
        objectOnly(repmat(~mask, [1, 1, 3])) = 0; % Set non-object regions to black

        volume3D(:, :, :, volumeIdx) = objectOnly;

        volumeIdx = volumeIdx + 1 + blankSlices;
        frameIdx = frameIdx + 1;
    end

    disp(['3D volume shape with spacing: ', num2str(size(volume3D))]);

    % Rearrange dimensions to match volshow requirements (one line. 30 minutes of debugging)
    volume3D = permute(volume3D, [1, 2, 4, 3]);

    % Visualization
    if exist('volshow', 'file')
        fig = uifigure('Name', '3D Volume Viewer', 'Position', [100 100 800 600]);
        volshow(volume3D, 'Parent', fig);
        uiwait(fig); %(wait so it doesnt close)
    else
        figure;
        for i = 1:size(volume3D, 3)
            imshow(squeeze(volume3D(:, :, i, :)), []);
            title(['Slice ', num2str(i)]);
            pause(0.1);
        end
    end

end

