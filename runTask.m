function runTask(varargin)
%% Parameters
argParser = inputParser();
argParser.KeepUnmatched = true;
argParser.addParameter('dataPath', fileparts(mfilename('fullpath')), ...
    @(p) exist(p, 'dir'));
argParser.addParameter('kfoldValues', [], @(x) ~isempty(x) && isnumeric(x));
argParser.addParameter('kfold', 5, @isnumeric);
argParser.addParameter('getRows', [], @(f) isa(f, 'function_handle'));
argParser.addParameter('getLabels', [], @(f) isa(f, 'function_handle'));
argParser.addParameter('featureExtractors', {}, ...
    @(fs) iscell(fs) && ~isempty(fs) ...
    && all(cellfun(@(f) isa(f, 'FeatureExtractor'), fs)));
argParser.addParameter('classifier', @LibsvmClassifierCCV, ...
    @(c) isa(c, 'function_handle'));
argParser.addParameter('resultsFilename', ...
    [datestr(datetime(), 'yyyy-mm-dd_HH-MM-SS'), '.mat'], @ischar);

argParser.parse(varargin{:});
dataPath = argParser.Results.dataPath;
kfoldValues = argParser.Results.kfoldValues;
kfold = argParser.Results.kfold;
getRows = argParser.Results.getRows;
getLabels = argParser.Results.getLabels;
featureExtractors = argParser.Results.featureExtractors;
assert(~isempty(featureExtractors), 'featureExtractors must not be empty');
classifierConstructor = argParser.Results.classifier;
resultsFilename = argParser.Results.resultsFilename;

%% Setup
% classifiers
classifiers = cellfun(@(featureExtractor) ...
    classifierConstructor(featureExtractor), ...
    featureExtractors, 'UniformOutput', false);
assert(all(cellfun(@(c) isa(c, 'Classifier'), classifiers)), ...
    'classifier must be of type ''Classifier''');
% data
% cross validation
rng(1, 'twister'); % seed, use pseudo random generator for reproducibility

%% Run
evaluateClassifiers = curry(@evaluate, classifiers, getRows, getLabels);
% parallelPoolObject = parpool; % init parallel computing pool
% crossValStream = RandStream('mlfg6331_64');
% reset(crossValStream);
results = crossval(evaluateClassifiers, kfoldValues, 'kfold', kfold);%, ...
%     'Options', statset('UseParallel', true, ...
%     'Streams', crossValStream, 'UseSubstreams', true));
% delete(parallelPoolObject); % teardown pool
resultsFile = [dataPath, '/results/' resultsFilename];
save(resultsFile, 'results');
fprintf('Results stored in ''%s''\n', resultsFile);
end
