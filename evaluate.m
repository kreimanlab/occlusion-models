function results = evaluate(classifiers, getRows, getLabels, ...
    train, test)
trainRows = getRows(train, RunType.Train);
trainLabels = getLabels(trainRows);
testRows = getRows(test, RunType.Test);
testLabels = getLabels(testRows);
results = cell(numel(classifiers), 1);
parfor iClassifier = 1:numel(classifiers)
    classifier = classifiers{iClassifier};
    fprintf('Training %s...\n', classifier.getName());
    classifier.train(trainRows, trainLabels);
    fprintf('Testing %s...\n', classifier.getName());
    predictedLabels = classifier.predict(testRows);
    % analyze
    correct = analyzeResults(predictedLabels, testLabels);
    currentResults = struct2dataset(struct(...
        'name', {repmat({classifier.getName()}, length(testRows), 1)}, ...
        'testrows', testRows', ...
        'response', predictedLabels, 'truth', testLabels, ...
        'correct', correct));
    results(iClassifier) = {currentResults};
end
% merge datasets and box for encapsulating cross validation
results = {vertcat(results{:})};
end