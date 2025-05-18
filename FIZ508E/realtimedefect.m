if ~isfile('trainedDefNet.mat')
        url = 'https://www.mathworks.com/supportfiles/dlhdl/trainedDefNet.mat';
        websave('trainedDefNet.mat',url);
    end
    net1 = load('trainedDefNet.mat');
    snet_defnet = net1.net;
    analyzeNetwork(net1.net);
%% Setup

if ~isfile('trainedBlemDetNet.mat')
        url = 'https://www.mathworks.com/supportfiles/dlhdl/trainedBlemDetNet.mat';
        websave('trainedBlemDetNet.mat',url);
    end
    net2 = load('trainedBlemDetNet.mat');
    snet_blemdetnet = net2.net
        analyzeNetwork(snet_blemdetnet)
%% cam = webcam;
cam = webcam,

%% Loop for real-time prediction
% Get input image size requirements for the networks
inputSize_defnet = snet_defnet.Layers(1).InputSize;
inputSize_blemdetnet = snet_blemdetnet.Layers(1).InputSize;

%% Create figure for display
figure;
if isa(snet_defnet, 'dlnetwork')
    defnet_predict = @(x) extractdata(predict(snet_defnet, dlarray(x, 'SSCB')));
else
    defnet_predict = @(x) predict(snet_defnet, x);
end

if isa(snet_blemdetnet, 'dlnetwork')
    blemdet_predict = @(x) extractdata(predict(snet_blemdetnet, dlarray(x, 'SSCB')));
else
    blemdet_predict = @(x) predict(snet_blemdetnet, x);
end

% Main loop for real-time prediction
try
    while true
        % Capture frame from webcam
        img = snapshot(cam);
        
        % Process for defect network
        img_defnet = imresize(img, inputSize_defnet(1:2));
        
        % Convert to correct format if needed (RGB or grayscale depending on network input)
        if inputSize_defnet(3) == 1 && size(img_defnet, 3) == 3
            img_defnet = rgb2gray(img_defnet);
        elseif inputSize_defnet(3) == 3 && size(img_defnet, 3) == 1
            img_defnet = repmat(img_defnet, [1, 1, 3]);
        end
        
        % Normalize image if needed (0-1 range)
        if isfloat(img_defnet) == false
            img_defnet = im2single(img_defnet);
        end
        
        defnet_pred = defnet_predict(img_defnet);
        [maxVal_defnet, defnet_idx] = max(defnet_pred);
        defnet_labels = {'Normal', 'Defect'}; % Update with your actual class names
        defnet_result = defnet_labels{defnet_idx};
        
        
        % Process for blemish detection network
        img_blemdetnet = imresize(img, inputSize_blemdetnet(1:2));
        
        % Convert to correct format if needed (RGB or grayscale depending on network input)
        if inputSize_blemdetnet(3) == 1 && size(img_blemdetnet, 3) == 3
            img_blemdetnet = rgb2gray(img_blemdetnet);
        elseif inputSize_blemdetnet(3) == 3 && size(img_blemdetnet, 3) == 1
            img_blemdetnet = repmat(img_blemdetnet, [1, 1, 3]);
        end
        
        % Normalize image if needed (0-1 range)
        if isfloat(img_blemdetnet) == false
            img_blemdetnet = im2single(img_blemdetnet);
        end
        
        blemdet_pred = blemdet_predict(img_blemdetnet);
        [maxVal_blemdet, blemdet_idx] = max(blemdet_pred);
        blemdet_labels = {'No Blemish', 'Blemish Detected'}; % Update with your actual class names
        blemdet_result = blemdet_labels{blemdet_idx};
        
        % Display results
        subplot(1,2,1), imshow(img);
        title(['Defect Network: ' defnet_result ' (' num2str(maxVal_defnet*100, '%.1f') '%)']);
        
        subplot(1,2,2), imshow(img);
        title(['Blemish Network: ' blemdet_result ' (' num2str(maxVal_blemdet*100, '%.1f') '%)']);
        
        drawnow;
        
        % Optional: Add a pause to control frame rate
        pause(0.1);
        
        % Check if figure is closed to exit loop
        if ~ishandle(gcf)
            break;
        end
    end
catch e
    disp(['Error: ' e.message]);
end
