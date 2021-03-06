function retrainAlexnet(predictOnly, predictSubdirectory)
if ~exist('predictOnly', 'var') || ~predictOnly
    %% Data
    fprintf('Loading data\n');
    images = imageDatastore('data/images',...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');
    
    %% Network
    fprintf('Creating network\n');
    net = alexnet;
    fc7Layers = net.Layers(1:end - 3);
    numClasses = numel(unique(images.Labels));
    layers = [
        fc7Layers
        fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
        softmaxLayer
        classificationLayer];
    
    %% Train
    crossValidations = 5;
    partitions = repmat(numel(images.Files) / crossValidations / numClasses, [crossValidations, 1]);
    partitions = mat2cell(partitions, ones(1, numel(partitions)));
    [p1, p2, p3, p4, p5] = splitEachLabel(images, partitions{:});
    partitions = {p1, p2, p3, p4, p5};
    for crossValidation = 1:crossValidations
        trainingPartitions = 1:crossValidations;
        trainingPartitions(crossValidation) = [];
        fprintf('Training on %s\n', mat2str(trainingPartitions));
        devData = partitions(trainingPartitions);
        devData = mergeImageDatastores(devData);
        [trainingData, validationData] = splitEachLabel(devData, 0.7, 'randomized');
        testData = partitions{crossValidation};
        
        miniBatchSize = 10;
        numIterationsPerEpoch = floor(numel(trainingData.Labels) / miniBatchSize);
        options = trainingOptions('sgdm',...
            'MiniBatchSize', miniBatchSize,...
            'MaxEpochs', 100,...
            'InitialLearnRate', 1e-4,...
            'Verbose', false,...
            'Plots','training-progress',...
            'ValidationData', validationData,...
            'ValidationFrequency', numIterationsPerEpoch);
        fineTrainedNet = trainNetwork(trainingData, layers, options);
        testFiles = testData.Files;
        save([fileparts(mfilename('fullpath')), '/alexnet-retrain-', num2str(crossValidation), '.mat'], ...
            'fineTrainedNet', 'testFiles', '-v7.3');
    end
end

fprintf('Predicting\n');
if ~exist('predictSubdirectory', 'var')
    predictSubdirectory = 'occluded';
end
predictAll(predictSubdirectory);
end

function predictAll(subDirectory)
layer = 'relu7';
crossValidations = 5;
features = NaN(0, 4096); % TODO: get rid of hard-coding, make dynamic
for crossValidation = 1:crossValidations
    %% load
    loaded = load([fileparts(mfilename('fullpath')), '/alexnet-retrain-', num2str(crossValidation), '.mat']);
    fineTrainedNet = loaded.fineTrainedNet;
    testFiles = loaded.testFiles;
    imagePaths = cell(0);
    indices = NaN(0);
    for f = 1:numel(testFiles)
        filePattern = strrep(testFiles{f}, '\', '/');
        filePattern = strrep(filePattern, '/images/', ...
            ['/images/', subDirectory, '/']);
        if ~isempty(subDirectory)
            filePattern = strrep(filePattern, '.png', '-*.png');
        end
        paths = dir(filePattern);
        for p = 1:size(paths, 1)
            basename = paths(p).name;
            path = strcat(paths(p).folder, '/', basename);
            imagePaths(end + 1) = {path};
            if ~isempty(subDirectory)
                index = strsplit(basename, '-');
                index = index{end};
            else
                index = basename;
            end
            index = strsplit(index, '.');
            indices(end + 1) = str2num(index{1});
        end
    end
    assert(numel(imagePaths) > 0);
    %% create data store
    testData = imageDatastore(imagePaths);
    assert(numel(testData.Files) == numel(imagePaths));
    for f = 1:numel(imagePaths)
        assert(strcmp(strrep(testData.Files{f}, '\', '/'), ...
            strrep(imagePaths{f}, '\', '/'))); % verify ordering
    end
    %% Predict
    fprintf('Predicting features %d/%d (%d images)\n', ...
        crossValidation, crossValidations, numel(testData.Files));
    testFeatures = activations(fineTrainedNet, testData, layer);
    features(indices, :) = testFeatures;
end
saveFile = ['data/features/data_occlusion_klab325v2/alexnet-retrain-', ...
    layer, '-', subDirectory, '.mat'];
fprintf('Saving to %s\n', saveFile);
save(saveFile, 'features', '-v7.3');
end

function merged = mergeImageDatastores(mergeInputs)
[merged, dummy] = splitEachLabel(mergeInputs{1}, 1);
mergedFiles = [merged.Files; dummy.Files];
mergedLabels = [merged.Labels; dummy.Labels];
merged.Files = mergedFiles;
merged.Labels = mergedLabels;
% concatenate cells in the new image datastores
for i = 2:numel(mergeInputs)
    mergedFiles = [merged.Files; mergeInputs{i}.Files];
    mergedLabels = [merged.Labels; mergeInputs{i}.Labels];
    merged.Files = mergedFiles;
    merged.Labels = mergedLabels;
end
end
