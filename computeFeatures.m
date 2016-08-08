function computeFeatures(varargin)
%% Setup
argParser = inputParser();
argParser.KeepUnmatched = true;
argParser.addParameter('data', [], @(d) isa(d, 'dataset'));
argParser.addParameter('dataSelection', [], @isnumeric);
argParser.addParameter('splitSize', 2000, @(x) isnumeric(x) && x > 0);
argParser.addParameter('images', [], @(i) iscell(i) && ~isempty(i));
argParser.addParameter('objectForRow', [], @(i) isnumeric(i) && ~isempty(i));
argParser.addParameter('numBubbles', [], @(i) isnumeric(i) && ~isempty(i));
argParser.addParameter('bubbleCenters', [], @(i) isnumeric(i) && ~isempty(i));
argParser.addParameter('bubbleSigmas', [], @(i) isnumeric(i) && ~isempty(i));
argParser.addParameter('trainDirectory', [], @(p) exist(p, 'dir'));
argParser.addParameter('testDirectory', [], @(p) exist(p, 'dir'));
argParser.addParameter('featureExtractors', {}, ...
    @(fs) iscell(fs) && all(cellfun(@(f) isa(f, 'FeatureExtractor'), fs)));
argParser.addParameter('masked', false, @(b) b == true || b == false);

argParser.parse(varargin{:});
fprintf('Computing features in %s with args:\n', pwd);
disp(argParser.Results);
data = argParser.Results.data;
dataSelection = argParser.Results.dataSelection;
splitSize = argParser.Results.splitSize;
images = argParser.Results.images;
objectForRow = argParser.Results.objectForRow;
numsBubbles = argParser.Results.numBubbles;
bubbleCenters = argParser.Results.bubbleCenters;
bubbleSigmas = argParser.Results.bubbleSigmas;
trainDir = argParser.Results.trainDirectory;
testDir = argParser.Results.testDirectory;
featureExtractors = argParser.Results.featureExtractors;
assert(~isempty(featureExtractors), 'featureExtractors must not be empty');
maskImages = argParser.Results.masked;

%% Run
parallelPoolObject = parpool; % init parallel computing pool
[~, uniquePresRows] = unique(data.pres);
numWhole = numel(uniquePresRows);
numWholeSplits = ceil(numWhole / splitSize);
numOccluded = size(data, 1);
numOccludedSplits = ceil(numOccluded / splitSize);
for featureExtractorIter = 1:length(featureExtractors)
    featureExtractor = featureExtractors{featureExtractorIter};
    if ~maskImages
        featureExtractor = ImageProvider(featureExtractor, images, ...
            objectForRow, numsBubbles, bubbleCenters, bubbleSigmas);
    else
        spectra = meanSpectra(images);
        featureExtractor = MaskedImageProvider(featureExtractor, images, ...
            objectForRow, numsBubbles, bubbleCenters, bubbleSigmas, ...
            spectra);
    end
    
    % have to artificially offset size to comply with parfor
    wholeFeatures = cell(numWholeSplits + numOccludedSplits, 1);
    occludedFeatures = cell(numOccludedSplits, 1);
    parfor dataIter = 1:numOccludedSplits + numWholeSplits
        if dataIter > numOccludedSplits
            % whole
            wholeIter = dataIter - numOccludedSplits;
            dataStart = (wholeIter - 1) * splitSize + 1;
            dataEnd = min(dataStart + splitSize - 1, numWhole);
            fprintf('%s whole images %d/%d (%d:%d)\n', ...
                featureExtractor.getName(), ...
                wholeIter, numWholeSplits, dataStart, dataEnd);
            wholeFeatures{dataIter} = ... % offset-index
                featureExtractor.extractFeatures(...
                uniquePresRows(dataStart:dataEnd), RunType.Train, []);
        else
            % occluded
            dataStart = (dataIter - 1) * splitSize + 1;
            dataEnd = min(dataStart + splitSize - 1, numOccluded);
            fprintf('%s occluded images %d/%d (%d:%d)\n', ...
                featureExtractor.getName(), ...
                dataIter, numOccludedSplits, dataStart, dataEnd);
            occludedFeatures{dataIter} = featureExtractor.extractFeatures(...
                intersect(dataStart:dataEnd, dataSelection), ...
                RunType.Test, []);
        end
    end
    % save
    features = cell2mat(wholeFeatures);
    saveFeatures(features, trainDir, featureExtractor);
    features = cell2mat(occludedFeatures);
    saveFeatures(features, testDir, featureExtractor);
end
delete(parallelPoolObject); % teardown pool
end

function saveFeatures(features, dir, classifier)
saveFile = sprintf('%s/%s.mat', dir, classifier.getName());
save(saveFile, '-v7.3', 'features');
end
