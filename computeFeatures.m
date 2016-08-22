function computeFeatures(varargin)
%% Setup
argParser = inputParser();
argParser.KeepUnmatched = true;
argParser.addParameter('data', [], @(d) isa(d, 'dataset'));
argParser.addParameter('dataSelection', [], @isnumeric);
argParser.addParameter('splitSize', 2000, @(x) isnumeric(x) && x > 0);
argParser.addParameter('images', [], @(i) iscell(i) && ~isempty(i));
argParser.addParameter('objectForRow', [], @(i) isnumeric(i) && ~isempty(i));
argParser.addParameter('adjustTestImages', [], @(f) isa(f, 'function_handle'));
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
adjustTestImages = argParser.Results.adjustTestImages;
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
    if maskImages
        averageSpectra = meanSpectra(...
            adjustTestImages(images, 1:numel(images)));
        adjustTestImages = @(images, ~) createPhaseScramble(...
                    size(images{1}), averageSpectra);
    end
    featureExtractor = ImageProvider(featureExtractor, images, ...
        objectForRow, adjustTestImages);
    
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
    fprintf('%s merge & save whole features...\n', ...
        featureExtractor.getName());
    features = cell2mat(wholeFeatures);
    saveFeatures(features, trainDir, featureExtractor);
    fprintf('%s merge & save occluded features...\n', ...
        featureExtractor.getName());
    features = cell2mat(occludedFeatures);
    saveFeatures(features, testDir, featureExtractor);
    fprintf('%s done.\n', featureExtractor.getName());
end
delete(parallelPoolObject); % teardown pool
end

function saveFeatures(features, dir, classifier)
saveFile = sprintf('%s/%s.mat', dir, classifier.getName());
save(saveFile, '-v7.3', 'features');
end
