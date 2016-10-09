function computeFeatures(varargin)
%% Setup
argParser = inputParser();
argParser.KeepUnmatched = true;
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
argParser.addParameter('omitWhole', false, @(b) b == true || b == false);
argParser.addParameter('omitOccluded', false, @(b) b == true || b == false);

argParser.parse(varargin{:});
fprintf('Computing features in %s with args:\n', pwd);
disp(argParser.Results);
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
omitWhole = argParser.Results.omitWhole;
omitOccluded = argParser.Results.omitOccluded;

%% Run
%parallelPoolObject = parpool; % init parallel computing pool
[~, uniquePresRows] = unique(objectForRow);
numWhole = numel(uniquePresRows);
numWholeSplits = ceil(numWhole / splitSize);
numOccluded = numel(objectForRow);
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
    
    %% compute
    %parfor dataIter = 1:numOccludedSplits + numWholeSplits
    for dataIter = 1:numOccludedSplits + numWholeSplits
        if dataIter > numOccludedSplits && ~omitWhole
            % whole
            wholeIter = dataIter - numOccludedSplits;
            dataStart = (wholeIter - 1) * splitSize + 1;
            dataEnd = min(dataStart + splitSize - 1, numWhole);
            saveFilename = getSaveFilename(featureExtractor, ...
                dataStart:dataEnd, numel(objectForRow));
            if exist(saveFilename, 'file') == 2
                fprintf('%s whole images skip %d/%d (file %s exists)\n', ...
                    featureExtractor.getName(), wholeIter, ...
                    numWholeSplits, saveFilename);
                continue;
            end
            rows = intersect(dataStart:dataEnd, dataSelection);
            if isempty(rows)
                fprintf('%s whole images skip %d/%d (empty)\n', ...
                    featureExtractor.getName(), wholeIter, numWholeSplits);
                continue;
            end
            fprintf('%s whole images %d/%d (%d:%d)\n', ...
                featureExtractor.getName(), ...
                dataIter, numWholeSplits, min(rows), max(rows));
            features = featureExtractor.extractFeatures(...
                uniquePresRows(rows), RunType.Train, []);
            saveFeatures(features, trainDir, saveFilename);
        elseif ~omitOccluded
            % occluded
            dataStart = (dataIter - 1) * splitSize + 1;
            dataEnd = min(dataStart + splitSize - 1, numOccluded);
            saveFilename = getSaveFilename(featureExtractor, ...
                dataStart:dataEnd, numel(objectForRow));
            if exist(saveFilename, 'file') == 2
                fprintf('%s whole images skip %d/%d (file %s exists)\n', ...
                    featureExtractor.getName(), wholeIter, ...
                    numWholeSplits, saveFilename);
                continue;
            end
            rows = intersect(dataStart:dataEnd, dataSelection);
            if isempty(rows)
                fprintf('%s occluded images skip %d/%d (empty)\n', ...
                    featureExtractor.getName(), dataIter, numOccludedSplits);
                continue;
            end
            fprintf('%s occluded images %d/%d (%d:%d)\n', ...
                featureExtractor.getName(), ...
                dataIter, numOccludedSplits, min(rows), max(rows));
            features = featureExtractor.extractFeatures(...
                rows, RunType.Test, []);
            saveFeatures(features, testDir, saveFilename);
        end
    end
    
    %% merge & save
    % whole
    if ~omitWhole
        fprintf('%s merge & save whole features...\n', ...
            featureExtractor.getName());
        features = cell(numWholeSplits, 1);
        for dataIter = 1:numWholeSplits
            dataStart = (dataIter - 1) * splitSize + 1;
            dataEnd = min(dataStart + splitSize - 1, numWhole);
            saveFilename = getSaveFilename(featureExtractor, ...
                dataStart:dataEnd, numel(objectForRow));
            features{dataIter} = loadFeatures(trainDir, saveFilename);
        end
        features = cell2mat(features);
        saveFilename = getSaveFilename(featureExtractor, dataSelection, numel(objectForRow));
        saveFeatures(features, trainDir, saveFilename);
    end
    
    % occluded
    if ~omitOccluded
        fprintf('%s merge & save occluded features...\n', ...
            featureExtractor.getName());
        features = cell(numOccludedSplits, 1);
        for dataIter = 1:numOccludedSplits
            dataStart = (dataIter - 1) * splitSize + 1;
            dataEnd = min(dataStart + splitSize - 1, numOccluded);
            saveFilename = getSaveFilename(featureExtractor, ...
                dataStart:dataEnd, numel(objectForRow));
            features{dataIter} = loadFeatures(testDir, saveFilename);
        end
        features = cell2mat(features);
        saveFilename = getSaveFilename(featureExtractor, dataSelection, numel(objectForRow));
        saveFeatures(features, testDir, saveFilename);
    end
    fprintf('%s done.\n', featureExtractor.getName());
end
%delete(parallelPoolObject); % teardown pool
end

function saveFeatures(features, dir, name)
saveFile = sprintf('%s/%s.mat', dir, name);
save(saveFile, '-v7.3', 'features');
end

function filename = getSaveFilename(featureExtractor, dataSelection, numRows)
filename = featureExtractor.getName();
if numel(dataSelection) ~= numRows
    filename = sprintf('%s-%d_%d', ...
        filename, min(dataSelection), max(dataSelection));
end
end

function features = loadFeatures(dir, name)
saveFile = sprintf('%s/%s.mat', dir, name);
features = load(saveFile);
features = features.features;
end
