function prepareData()
directory = [fileparts(mfilename('fullpath')), '/images'];
dataFilepath = [directory, '/data.txt'];
if exist(dataFilepath, 'file') == 2
    delete(dataFilepath);
end
mkdir(directory);
imageSize = 227;

images = load('data/KLAB325.mat');
images = images.img_mat;
occlusionData = load('data/data_occlusion_klab325v2.mat');
occlusionData = occlusionData.data;
bubbleSigmas = repmat(14, [size(occlusionData, 1), 10]);

wholeFeatures = load('data/features/klab325_orig/alexnet-relu7.mat');
wholeFeatures = wholeFeatures.features;

fileID = fopen(dataFilepath, 'w');

for i = 1:325
    %% whole
    baseImage = images{i};
    wholeFeats = wholeFeatures(i, :);
    %% occluded
    occlusionDataSelection = find(occlusionData.pres == i)';
    assert(numel(unique(occlusionData.truth(occlusionDataSelection))) == 1);
    for row = occlusionDataSelection
        occludedImage = occlude({baseImage}, occlusionData.nbubbles(row), ...
            occlusionData.bubble_centers(row, :), bubbleSigmas(row, :));
        occludedImage = convertImage(occludedImage{1}, imageSize);
        filepath = [directory, '/', sprintf('%d', row), '.png'];
        imwrite(occludedImage, filepath);
        %% write
        unixFilepath = strrep(strrep(filepath, '\', '/'), 'C:/', '/mnt/c/');
        fprintf(fileID, '%s ', unixFilepath);
        fprintf(fileID, '%f,', wholeFeats(1:end-1));
        fprintf(fileID, '%f', wholeFeats(end));
        fprintf(fileID, '\n');
    end
end

fclose(fileID);
end

function image = convertImage(baseImage, imageSize)
image = imresize(baseImage, [imageSize, imageSize]);
image = grayscaleToRgb(image, 'channels-last');
end