
function defectDetectionGUI
    % Defect and Blemish detection using pretrained networks with GUI
    
    % Main figure
    fig = figure('Name', 'Defect Detection System', 'Position', [100, 100, 1000, 600], ...
                'NumberTitle', 'off', 'MenuBar', 'none', 'Resize', 'on');
    
    % UI components (GUI)
    hPanel = uipanel('Parent', fig, 'Title', 'Image Input', ...
                     'Position', [0.02, 0.7, 0.3, 0.25]);
    
    uicontrol('Parent', hPanel, 'Style', 'text', 'String', 'Image Path:', ...
              'Position', [10, 80, 80, 25], 'HorizontalAlignment', 'left');
    
    pathEdit = uicontrol('Parent', hPanel, 'Style', 'edit', ...
                         'Position', [90, 80, 180, 25]);
    
    browseBtn = uicontrol('Parent', hPanel, 'Style', 'pushbutton', ...
                         'String', 'Browse...', 'Position', [10, 40, 80, 25], ...
                         'Callback', @browseCallback);
    
    analyzeBtn = uicontrol('Parent', hPanel, 'Style', 'pushbutton', ...
                         'String', 'Analyze', 'Position', [100, 40, 80, 25], ...
                         'Callback', @analyzeCallback);
    
    % Panels for displaying images and results
    origImgPanel = uipanel('Parent', fig, 'Title', 'Original Image', ...
                          'Position', [0.02, 0.1, 0.3, 0.55]);
    origImgAx = axes('Parent', origImgPanel);
    
    defnetPanel = uipanel('Parent', fig, 'Title', 'Defect Network Results', ...
                         'Position', [0.35, 0.1, 0.3, 0.85]);
    defnetAx = axes('Parent', defnetPanel);
    
    blemnetPanel = uipanel('Parent', fig, 'Title', 'Blemish Detection Results', ...
                          'Position', [0.68, 0.1, 0.3, 0.85]);
    blemnetAx = axes('Parent', blemnetPanel);
    
    % Status panel
    statusPanel = uipanel('Parent', fig, 'Title', 'Status', ...
                         'Position', [0.35, 0.7, 0.3, 0.25]);
    statusText = uicontrol('Parent', statusPanel, 'Style', 'text', ...
                          'Position', [10, 10, 280, 80], ...
                          'HorizontalAlignment', 'left', ...
                          'String', 'Ready. Please select an image file.');
    
    % Loading the neural networks
    setStatus('Loading neural networks. Please wait...');
    
    % Store variables in app data
    appData = struct();
    appData.pathEdit = pathEdit;
    appData.origImgAx = origImgAx;
    appData.defnetAx = defnetAx;
    appData.blemnetAx = blemnetAx;
    appData.statusText = statusText;
    
    % Load neural networks
    try
        % Load Defect Detection Network
        if ~isfile('trainedDefNet.mat')
            url = 'https://www.mathworks.com/supportfiles/dlhdl/trainedDefNet.mat';
            websave('trainedDefNet.mat', url);
        end
        net1 = load('trainedDefNet.mat');
        appData.snet_defnet = net1.net;
        
        % Load Blemish Detection Network
        if ~isfile('trainedBlemDetNet.mat')
            url = 'https://www.mathworks.com/supportfiles/dlhdl/trainedBlemDetNet.mat';
            websave('trainedBlemDetNet.mat', url);
        end
        net2 = load('trainedBlemDetNet.mat');
        appData.snet_blemdetnet = net2.net;

        % Input size requirements for the networks
        inputLayerIdx1 = find(arrayfun(@(x) isa(x, 'nnet.cnn.layer.ImageInputLayer'), appData.snet_defnet.Layers));
        if ~isempty(inputLayerIdx1)
            appData.inputSize_defnet = appData.snet_defnet.Layers(inputLayerIdx1).InputSize;
        else
            % For dlnetworks, try to infer the input size
            appData.inputSize_defnet = [224, 224, 3]; % Default size if unable to determine
        end
        
        inputLayerIdx2 = find(arrayfun(@(x) isa(x, 'nnet.cnn.layer.ImageInputLayer'), appData.snet_blemdetnet.Layers));
        if ~isempty(inputLayerIdx2)
            appData.inputSize_blemdetnet = appData.snet_blemdetnet.Layers(inputLayerIdx2).InputSize;
        else
            % For dlnetworks, try to infer the input size
            appData.inputSize_blemdetnet = [224, 224, 3]; % Default size if unable to determine
        end
        
        setStatus('Neural networks loaded successfully. Ready to analyze images.');
    catch ME
        setStatus(['Error loading networks: ', ME.message]);
    end
    
    % Store app data
    guidata(fig, appData);
    
    % Nested functions for callbacks
    function browseCallback(~, ~)
        [filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp', 'Image Files (*.jpg, *.png, *.bmp)'}, 'Select an image');
        if filename ~= 0
            fullpath = fullfile(pathname, filename);
            set(pathEdit, 'String', fullpath);
        end
    end
    
    function analyzeCallback(~, ~)
        appData = guidata(fig);
        imagePath = get(appData.pathEdit, 'String');
        
        if isempty(imagePath) || ~isfile(imagePath)
            setStatus('Invalid image path. Please select a valid image file.');
            return;
        end
        
        setStatus(['Processing image: ', imagePath]);
        
        try
            % Load and display original image
            img = imread(imagePath);
            axes(appData.origImgAx);
            imshow(img);
            title('Original Image', 'FontSize', 12, 'FontWeight', 'bold');
            
            % Process with first network (defect network)
            setStatus('Running defect detection network...');
            [defnetResult, defectLabel, defectScore] = processWithDLNetwork(img, appData.snet_defnet, appData.inputSize_defnet, 'Defect');
            axes(appData.defnetAx);
            imshow(defnetResult);
            
            % Add label on the image with clear background
            % Create a more visible and stable label by adding a colored banner at the top
            hold on;
            width = size(defnetResult, 2);
            bannerHeight = 40;
            rectangle('Position', [0, 0, width, bannerHeight], 'FaceColor', 'red', 'EdgeColor', 'none');
            labelText = 'Defect Prediction';
            if ~isempty(defectLabel)
                labelText = [labelText ': ' defectLabel];
                if defectScore >= 0
                    labelText = [labelText ' (' num2str(defectScore, '%.2f') ')'];
                end
            end
            text(width/2, bannerHeight/2, labelText, 'Color', 'white', 'FontWeight', 'bold', ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 12);
            hold off;
            
            % Process with second network (blemish detection network)
            setStatus('Running blemish detection network...');
            [blemnetResult, blemishLabel, blemishScore] = processWithDLNetwork(img, appData.snet_blemdetnet, appData.inputSize_blemdetnet, 'Blemish');
            axes(appData.blemnetAx);
            imshow(blemnetResult);
            
            % Add label on the image with clear background
            hold on;
            width = size(blemnetResult, 2);
            bannerHeight = 40;
            rectangle('Position', [0, 0, width, bannerHeight], 'FaceColor', 'blue', 'EdgeColor', 'none');
            labelText = 'Blemish Prediction';
            if ~isempty(blemishLabel)
                labelText = [labelText ': ' blemishLabel];
                if blemishScore >= 0
                    labelText = [labelText ' (' num2str(blemishScore, '%.2f') ')'];
                end
            end
            text(width/2, bannerHeight/2, labelText, 'Color', 'white', 'FontWeight', 'bold', ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 12);
            hold off;
            
            setStatus('Analysis complete!');
        catch ME
            setStatus(['Error during analysis: ', ME.message]);
            disp(getReport(ME, 'extended'));
        end
    end
    
    function setStatus(message)
        set(statusText, 'String', message);
        drawnow;
    end
end

function [resultImg, predLabel, maxScore] = processWithDLNetwork(img, network, inputSize, networkType)
    % Process image with a dlnetwork
    
    % Default return values
    predLabel = '';
    maxScore = -1;
    
    % Resize image to network input size while preserving color
    img_resized = imresize(img, [inputSize(1:2)]);
    
    % Handle channel requirements (RGB to RGB) without grayscale conversion
    % If network expects 3 channels and image has only 1, duplicate the single channel
    if size(img_resized, 3) == 1 && inputSize(3) == 3
        img_resized = cat(3, img_resized, img_resized, img_resized);
    end
    % If network expects 1 channel and image has 3, take the first channel
    % but preserve this as a color image for visualization
    imgForVis = img_resized; % Keep original for visualization
    if size(img_resized, 3) == 3 && inputSize(3) == 1
        imgForNetwork = rgb2gray(img_resized);
    else
        imgForNetwork = img_resized;
    end
    
    % Prepare for dlnetwork
    dlImg = dlarray(single(imgForNetwork), 'SSCB');
    
    % Run prediction using the dlnetwork
    try
        % Attempt to run the network
        result = predict(network, dlImg);
        
        % For classification networks
        if ~isempty(result) && (isnumeric(result) || isdlarray(result))
            if isdlarray(result)
                scores = extractdata(result);
            else
                scores = result;
            end
            
            % Handle different output formats
            if numel(scores) > 1
                % Multiple scores, get the highest
                [maxScore, predIdx] = max(scores(:));
                predLabel = ['Class ' num2str(predIdx)];
                
                % Try to get class names if available
                try
                    outputLayerIdx = find(arrayfun(@(x) isa(x, 'nnet.cnn.layer.ClassificationOutputLayer'), network.Layers));
                    if ~isempty(outputLayerIdx)
                        classNames = network.Layers(outputLayerIdx).Classes;
                        predLabel = classNames{predIdx};
                    end
                catch
                    % Just use the numeric class if names are not available
                end
                
            else
                % Binary classification
                maxScore = scores;
                if scores > 0.5
                    predLabel = 'Positive';
                else
                    predLabel = 'Negative';
                end
            end
            
            % Create visualization using the original color image
            resultImg = imgForVis;
            
            % For semantic segmentation type outputs
            if numel(scores) == numel(imgForNetwork)
                % Reshape scores to image size
                segMap = reshape(scores, size(imgForNetwork, 1), size(imgForNetwork, 2));
                segMap = rescale(segMap);
                
                % Create colored overlay on original color image
                heatmap = ind2rgb(gray2ind(segMap, 256), jet(256));
                
                % Blend original image with heatmap while keeping colors
                alpha = 0.4; % Transparency of heatmap
                resultImg = (1-alpha) * im2double(imgForVis) + alpha * heatmap;
            end
            
        else
            % Couldn't get meaningful prediction, return original image
            resultImg = imgForVis;
        end
    catch ME
        % Error in prediction, return original image
        resultImg = imgForVis;
        disp(['Error in ' networkType ' network: ' ME.message]);
    end
end