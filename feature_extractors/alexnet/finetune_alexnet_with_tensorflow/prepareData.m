function prepareData()
directory = [fileparts(mfilename('fullpath')), '/images'];
wholeDirectory = [directory, '/whole/'];
occludedDirectory = [directory, '/occluded/'];
mkdir(directory);
mkdir(wholeDirectory);
mkdir(occludedDirectory);
imageSize = 227;
kfolds = 5;
validationSplit = 0.1;

objects = 1:325;
images = load('data/KLAB325.mat');
images = images.img_mat;
occlusionData = load('data/data_occlusion_klab325v2.mat');
occlusionData = occlusionData.data;
bubbleSigmas = repmat(14, [size(occlusionData, 1), 10]);

wholeFeatures = load('data/features/klab325_orig/alexnet-relu7.mat');
wholeFeatures = wholeFeatures.features;

fprintf('Processing objects\n');
objectFilepaths = cell(size(objects));
wholeObjectFilepaths = cell(size(objects));
for i = objects
    fprintf('Object %d/%d\n', i, max(objects));
    baseImage = images{i};
    %% whole
    wholeImage = convertImage(baseImage, imageSize);
    filepath = [wholeDirectory, sprintf('%d', i), '.png'];
    imwrite(wholeImage, filepath);
    wholeObjectFilepaths = appendFilepath(filepath, wholeObjectFilepaths, i);
    %% occluded
    occlusionDataSelection = find(occlusionData.pres == i)';
    assert(numel(unique(occlusionData.truth(occlusionDataSelection))) == 1);
    for row = occlusionDataSelection
        %% occlude image
        occludedImage = occlude({baseImage}, occlusionData.nbubbles(row), ...
            occlusionData.bubble_centers(row, :), bubbleSigmas(row, :));
        occludedImage = convertImage(occludedImage{1}, imageSize);
        filepath = [occludedDirectory, sprintf('%d', row), '.png'];
        imwrite(occludedImage, filepath);
        objectFilepaths = appendFilepath(filepath, objectFilepaths, i);
    end
end

%% cross-validation
fprintf('Splitting for cross-validation\n');
crossfun = @(xtrainval, xtest) {xtrainval, xtest};
crossValidations = crossval(crossfun, objects', 'kfold', kfolds);
for kfold = 1:size(crossValidations, 1)
    fprintf('Kfold %d/%d\n', kfold, kfolds);
    % split
    trainValObjects = crossValidations{kfold, 1};
    [trainInd, valInd, ~] = dividerand(numel(trainValObjects), 1 - validationSplit, validationSplit, 0);
    trainObjects = trainValObjects(trainInd);
    valObjects = trainValObjects(valInd);
    testObjects = crossValidations{kfold, 2};
    % write to files
    trainFilepath = [directory, sprintf('/train%d.txt', kfold)];
    valFilepath = [directory, sprintf('/val%d.txt', kfold)];
    testFilepath = [directory, sprintf('/test%d.txt', kfold)];
    writeToFile(trainFilepath, trainObjects', wholeFeatures, objectFilepaths);
    writeToFile(valFilepath, valObjects', wholeFeatures, objectFilepaths);
    writeToFile(testFilepath, testObjects', wholeFeatures, objectFilepaths);
    writeToFile(testFilepath, testObjects', wholeFeatures, wholeObjectFilepaths, 'a');
end
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