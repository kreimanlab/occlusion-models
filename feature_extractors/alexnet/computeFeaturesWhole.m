function computeFeaturesWhole()

dir = fileparts(mfilename('fullpath'));
addpath([dir '/../data']);
netParams = load([dir '/ressources/alexnetParams.mat']);
imagesMean = load([dir '/ressources/ilsvrc_2012_mean.mat']);
imagesMean = imagesMean.mean_data;

dataSelection = 1:(325/5):325;

wholeImages = getWholeImages(dataSelection);
totalLength = length(wholeImages);
features = struct('pres', [], ...
    'c1', [], 'r1', [], 'p1', [], 'n1', [], ...
    'c2', [], 'r2', [], 'p2', [], 'n2', [], ...
    'c3', [], ...
    'c4', [], ...
    'c5', [], 'r5', [], 'p5', [], ...
    'fc6', [], 'r6', [], ...
    'fc7', [], 'r7', [], ...
    'fc8', [], ...
    'prob', []);
for i = 1:totalLength
    fprintf('%d/%d\n', i, totalLength);
    preparedImage = prepareGrayscaleImage(wholeImages{i}, imagesMean);
    layerOutputs = alexNetLayerOutputs(preparedImage, netParams);
    features(i).pres = dataSelection(i);
    for field = fieldnames(layerOutputs)'
        features(i).(field{1}) = layerOutputs.(field{1});
    end
end
selectionStr = num2str(dataSelection, '%d-');
save([dir '/../data/OcclusionModeling/features/klab325_orig/'...
    'alexnet_all_' selectionStr(1:end-1) '.mat'], ...
    '-v7.3', 'features');
