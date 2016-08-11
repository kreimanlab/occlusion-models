function computeHopTimeFeatures(varargin)

%% Setup
argParser = inputParser();
argParser.KeepUnmatched = true;
argParser.addParameter('objectForRow', [], @(x) ~isempty(x) && isnumeric(x));
argParser.addParameter('savesteps', [1:100, 110:10:300], @isnumeric);
argParser.addParameter('trainDirectory', [], @(p) exist(p, 'dir'));
argParser.addParameter('testDirectory', [], @(p) exist(p, 'dir'));
argParser.addParameter('weightsDirectory', [], @(p) exist(p, 'dir'));
argParser.addParameter('featureExtractor', [], ...
    @(f) isa(f, 'HopFeatures'));
argParser.addParameter('omitTrain', false, @(b) b == true || b == false);

argParser.parse(varargin{:});
fprintf('Computing features in %s with args:\n', pwd);
disp(argParser.Results);
objectForRow = argParser.Result.objectForRow;
savesteps = argParser.Results.savesteps;
trainDir = argParser.Results.trainDirectory;
testDir = argParser.Results.testDirectory;
weightsDir = argParser.Results.weightsDirectory;
featureExtractor = argParser.Results.featureExtractor;
omitTrain = argParser.Results.omitTrain;

%% whole
if ~omitTrain
    [~, wholePresRows] = unique(objectForRow);
    fprintf('Training on %d whole objects\n', numel(wholePresRows));
    features = featureExtractor.extractFeatures(wholePresRows, ...
        RunType.Train, []);
    net = featureExtractor.net;
    save([weightsDir, featureExtractor.getName(), '-net.mat'], 'net');
    for t = savesteps
        saveFeatures(features, trainDir, featureExtractor, t);
    end
end

%% occluded
sliceSize = 1000;
numSlices = ceil(size(objectForRow, 1) / sliceSize);
features = cell(numSlices, 1);
parfor sliceIter = 1:numSlices
    dataStart = (sliceIter - 1) * sliceSize + 1;
    dataEnd = dataStart + sliceSize - 1;
    fprintf('%s occluded %d/%d (%d:%d)\n', featureExtractor.getName(), ...
        sliceIter, numSlices, dataStart, dataEnd);
    rows = dataStart:dataEnd;
    [~, ys] = featureExtractor.extractFeatures(rows, RunType.Test, []);
    features{sliceIter} = ys(:, :, savesteps);
end
features = vertcat(features{:});
% save
parfor timeIter = 1:numel(savesteps)
    timestep = savesteps(timeIter);
    saveFeatures(features(:, :, timeIter), ...
        testDir, featureExtractor, timestep);
end
end

function saveFeatures(features, dir, featureExtractor, timestep)
% Replace last timestep with current timestep.
% Since matlab has no last option, replace the first of the reverse.
name = fliplr(regexprep(fliplr(featureExtractor.getName()), ...
    '[0-9]+t_', fliplr(['_t', num2str(timestep)]), 'once'));
save([dir, '/', name, '.mat'], '-v7.3', 'features');
end
