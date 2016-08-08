classdef FeatureProviderFactory < handle
    % Caches feature providers to minimize memory overhead.
    
    properties
        trainDirectory
        testDirectory
        objectForRow
        dataSelection
        
        featureProviders
    end
    
    methods
        function self = FeatureProviderFactory(...
                trainDirectory, testDirectory, ...
                objectForRow, dataSelection)
            self.trainDirectory = trainDirectory;
            self.testDirectory = testDirectory;
            self.objectForRow = objectForRow;
            self.dataSelection = dataSelection;
            self.featureProviders = containers.Map();
        end
        
        function featureProvider = get(self, originalExtractor)
            name = originalExtractor.getName();
            if isKey(self.featureProviders, name)
                featureProvider = self.featureProviders(name);
                return;
            end
            if strfind(originalExtractor.getName(), 'Rnn') == 1
                trainParentDir = realpath([self.trainDirectory, '..']);
                testParentDir = realpath([self.testDirectory, '..']);
                assert(strcmp(trainParentDir, testParentDir), ...
                    ['train and test parent directory do not match', ...
                    '(%s != %s)'], trainParentDir, testParentDir);
                featureProvider = RnnFeatureProvider(trainParentDir, ...
                    self.objectForRow, originalExtractor);
            else
                constructor = curry(@FeatureProvider, ...
                        self.trainDirectory, self.testDirectory, ...
                        self.objectForRow, self.dataSelection);
                if isa(originalExtractor, 'BipolarFeatures')
                    inputProvider = constructor(...
                        originalExtractor.featuresInput);
                    originalExtractor.featuresInput = inputProvider;
                    featureProvider = originalExtractor;
                else
                    featureProvider = constructor(...
                        originalExtractor);
                end
            end
            self.featureProviders(name) = featureProvider;
        end
        
        function remove(self, originalExtractor)
            name = originalExtractor.getName();
            if ~isKey(self.featureProviders, name)
                error('Unknown extractor %s', name);
            end
            remove(self.featureProviders, {name});
        end
    end
end
