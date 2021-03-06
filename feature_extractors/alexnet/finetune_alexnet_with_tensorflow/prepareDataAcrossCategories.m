function prepareDataAcrossCategories(occludedWholeRatio)
%PREPAREDATAACROSSCATEGORIES prepares the data for fine-tuning (mixed training)
%   occludedWholeRatio: number of occluded images to train on divided by 
%   number of whole images to train on

%% settings
if ~exist('occludedWholeRatio', 'var')
    occludedWholeRatio = 1/1;
end
imageSize = 227;
kfolds = 5;
validationSplit = 0.1;
rng(0);

%% directories
directory = [fileparts(mfilename('fullpath')), '/images-across_categories'];
wholeDirectory = [directory, '/whole/'];
occludedDirectory = [directory, '/occluded/'];
lessOccludedDirectory = [directory, '/lessOcclusion/'];
mkdir(directory);
mkdir(wholeDirectory);
mkdir(occludedDirectory);
mkdir(lessOccludedDirectory);

%% data
images = load('data/KLAB325.mat');
images = images.img_mat;
occlusionData = load('data/data_occlusion_klab325v2.mat');
occlusionData = occlusionData.data;
lessOcclusionData = load('data/lessOcclusion/data_occlusion_klab325-high_visibility.mat');
lessOcclusionData = lessOcclusionData.data;
bubbleSigmas = repmat(14, [size(occlusionData, 1), 10]);
wholeFeatures = load('data/features/klab325_orig/alexnet-relu7.mat');
wholeFeatures = wholeFeatures.features;
objects = unique(occlusionData.pres)';
categories = unique(occlusionData.truth);

%% draw samples
assert(occludedWholeRatio <= size(occlusionData, 1) / numel(objects));
wholeBin = objects;
occludedBin = 1:size(occlusionData, 1);
numOccludedDraws = round(occludedWholeRatio * numel(wholeBin));
numWholeDraws = min(round(numOccludedDraws / occludedWholeRatio), numel(wholeBin));
fprintf('Drawing %d occluded and %d whole images\n', numOccludedDraws, numWholeDraws);
wholeSamples = datasample(wholeBin, numWholeDraws, 'Replace', false);
occludedSamples = datasample(occludedBin, numOccludedDraws, 'Replace', false);
assert(numel(wholeSamples) / numel(occludedSamples) == occludedWholeRatio);

%% process images, write to disk
fprintf('Processing objects\n');
allOccludedFilepaths = cell(size(objects));
allWholeFilepaths = cell(size(objects));
allLessOccludedFilepaths = cell(size(objects));
devOccludedFilepaths = cell(size(objects));
devWholeFilepaths = cell(size(objects));
for i = objects
    fprintf('Object %d/%d\n', i, max(objects));
    baseImage = images{i};
    %% whole
    wholeImage = convertImage(baseImage, imageSize);
    filepath = [wholeDirectory, sprintf('%d', i), '.png'];
    imwrite(wholeImage, filepath);
    allWholeFilepaths = appendFilepath(filepath, allWholeFilepaths, i);
    if ismember(i, wholeSamples)
        devWholeFilepaths = appendFilepath(filepath, devWholeFilepaths, i);
    end
    %% occluded
    occlusionDataSelection = find(occlusionData.pres == i)';
    assert(numel(unique(occlusionData.truth(occlusionDataSelection))) == 1);
    for row = occlusionDataSelection
        % occlude image
        occludedImage = occlude({baseImage}, occlusionData.nbubbles(row), ...
            occlusionData.bubble_centers(row, :), bubbleSigmas(row, :));
        occludedImage = convertImage(occludedImage{1}, imageSize);
        filepath = [occludedDirectory, sprintf('%d', row), '.png'];
        imwrite(occludedImage, filepath);
        allOccludedFilepaths = appendFilepath(filepath, allOccludedFilepaths, i);
        if ismember(row, occludedSamples)
            devOccludedFilepaths = appendFilepath(filepath, devOccludedFilepaths, i);
        end
    end
    %% less occluded
    occlusionDataSelection = find(lessOcclusionData.pres == i)';
    assert(numel(unique(lessOcclusionData.truth(occlusionDataSelection))) == 1);
    for row = occlusionDataSelection
        % occlude image
        occludedImage = occlude({baseImage}, lessOcclusionData.nbubbles(row), ...
            lessOcclusionData.bubble_centers(row, :), lessOcclusionData.bubbleSigmas(row, :));
        occludedImage = convertImage(occludedImage{1}, imageSize);
        filepath = [lessOccludedDirectory, sprintf('%d', row), '.png'];
        imwrite(occludedImage, filepath);
        allLessOccludedFilepaths = appendFilepath(filepath, allLessOccludedFilepaths, i);
    end
end
assert(numel(devOccludedFilepaths) / numel(devWholeFilepaths) == occludedWholeRatio);

%% cross-validation
fprintf('Splitting for cross-validation\n');
crossfun = @(xtest, xtrainval) {xtest, xtrainval};
crossValidations = crossval(crossfun, categories, 'kfold', kfolds);
for kfold = 1:size(crossValidations, 1)
    fprintf('Kfold %d/%d\n', kfold, kfolds);
    % split
    trainValCategories = crossValidations{kfold, 2};
    testCategories = crossValidations{kfold, 1};
    trainValObjects = objectsFromCategory(trainValCategories, occlusionData);
    [trainInd, valInd, ~] = dividerand(numel(trainValObjects), ...
        1 - validationSplit, validationSplit, 0);
    trainObjects = trainValObjects(trainInd);
    valObjects = trainValObjects(valInd);
    testObjects = objectsFromCategory(testCategories, occlusionData);
    % write to files
    trainFilepath = [directory, sprintf('/train%d.txt', kfold)];
    valFilepath = [directory, sprintf('/val%d.txt', kfold)];
    testFilepath = [directory, sprintf('/test%d.txt', kfold)];
    writeToFile(trainFilepath, trainObjects', wholeFeatures, devOccludedFilepaths);
    writeToFile(trainFilepath, trainObjects', wholeFeatures, devWholeFilepaths, 'a');
    writeToFile(valFilepath, valObjects', wholeFeatures, devOccludedFilepaths);
    writeToFile(valFilepath, valObjects', wholeFeatures, devWholeFilepaths, 'a');
    writeToFile(testFilepath, testObjects', wholeFeatures, allOccludedFilepaths);
    writeToFile(testFilepath, testObjects', wholeFeatures, allWholeFilepaths, 'a');
    writeToFile(testFilepath, testObjects', wholeFeatures, allLessOccludedFilepaths, 'a');
end
end

function objects = objectsFromCategory(categories, data)
objects = unique(data.pres(ismember(data.truth, categories)));
end

function image = convertImage(baseImage, imageSize)
image = imresize(baseImage, [imageSize, imageSize]);
image = grayscaleToRgb(image, 'channels-last');
end

function objectFilepaths = appendFilepath(filepath, objectFilepaths, i)
    unixFilepath = strrep(strrep(filepath, '\', '/'), 'C:/', '/mnt/c/');
    filepaths = objectFilepaths{i};
    filepaths{end + 1} = unixFilepath;
    objectFilepaths{i} = filepaths;
end

function writeToFile(filepath, objects, features, objectFilepaths, fileaccess)
    if ~exist('fileaccess', 'var')
        fileaccess = 'w';
    end
    fileID = fopen(filepath, fileaccess);
    for object = objects
        feats = features(object, :);
        imageFilepaths = objectFilepaths{object};
        for imageFilepath = imageFilepaths
            fprintf(fileID, '%s ', imageFilepath{1});
            fprintf(fileID, '%f,', feats(1:end-1));
            fprintf(fileID, '%f', feats(end));
            fprintf(fileID, '\n');
        end
    end
    fclose(fileID);
end
