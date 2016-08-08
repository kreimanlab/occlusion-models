classdef FeatureProvider < FeatureExtractor
    % Classifier that retrieves the features from previous runs.
    
    properties
        originalExtractor
        caches
    end
    
    methods
        function self = FeatureProvider(trainDirectory, testDirectory, ...
                objectForRow, dataSelection, originalExtractor)
            self.originalExtractor = originalExtractor;
            [filename, loadFeatures] = ...
                self.getFileDirectives(originalExtractor);
            trainCache = self.createTrainCache(trainDirectory, ...
                filename, loadFeatures, objectForRow);
            testCache = self.createTestCache(testDirectory, ...
                filename, loadFeatures, dataSelection);
            self.caches = containers.Map(...
                {char(RunType.Train), char(RunType.Test)}, ...
                {trainCache, testCache});
        end
        
        function name = getName(self)
            name = self.originalExtractor.getName();
        end
        
        function features = extractFeatures(self, rows, runType, ~)
            boxedFeatures = cell(length(rows), 1);
            cache = self.caches(char(runType));
            for i = 1:length(rows)
                cachedFeatures = cache(rows(i));
                boxedFeatures{i} = cachedFeatures{:};
            end
            features = cell2mat(boxedFeatures);
        end
    end
    
    methods (Access=private)
        function cache = createTrainCache(~, ...
                dir, filename, ...
                loadFeatures, objectForRow)
            cache = containers.Map(...
                'KeyType', 'double', 'ValueType', 'any');
            filePath = [dir, filename];
            features = loadFeatures(filePath);
            for row = 1:size(objectForRow, 1)
                cache(row) = {features(objectForRow(row), :)};
            end
        end
        
        function cache = createTestCache(~, ...
                dir, filename, ...
                loadFeatures, dataSelection)
            cache = containers.Map(...
                'KeyType', 'double', 'ValueType', 'any');
            features = loadFeatures([dir, filename]);
            for id = 1:size(features, 1)
                if ~ismember(id, dataSelection)
                    continue;
                end
                cache(id) = {features(id,:)};
            end
        end
        
        function features = loadMat(~, filePath)
            data = load(filePath);
            features = data.features;
        end
        
        function [filename, loadFeatures] = ...
                getFileDirectives(self, originalExtractor)
            name = originalExtractor.getName();
            filename = [name, '.mat'];
            loadFeatures = @self.loadMat;
        end
    end
end
