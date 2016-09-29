classdef FeatureProviderFactory < handle
    % Caches feature providers to minimize memory overhead.
    
    properties
        trainDirectory
        testDirectory
        objectForRow
        dataSelection
        
        featureProviders
        
        images
        adjustTestImages
    end
    
    methods
        function self = FeatureProviderFactory(...
                trainDirectory, testDirectory, ...
                objectForRow, dataSelection, ...
                images, adjustTestImages)
            self.trainDirectory = self.appendSlash(trainDirectory);
            self.testDirectory = self.appendSlash(testDirectory);
            self.objectForRow = objectForRow;
            self.dataSelection = dataSelection;
            self.featureProviders = containers.Map();
            if ~exist('images', 'var')
                warning('no images provided for FeatureProviderFactory constructor');
                images = [];
            end
            self.images = images;
            if ~exist('adjustTestImages', 'var')
                warning('no adjustTestImages provided for FeatureProviderFactory constructor');
                adjustTestImages = [];
            end
            self.adjustTestImages = adjustTestImages;
        end
        
        function featureProvider = get(self, originalExtractor)
            name = originalExtractor.getName();
            if isKey(self.featureProviders, name)
                featureProvider = self.featureProviders(name);
                return;
            end
            if strfind(lower(originalExtractor.getName()), lower('Rnn')) == 1
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
                elseif isa(originalExtractor, 'PixelFeatures')
                    featureProvider = ImageProvider(originalExtractor, ...
                        self.images, self.objectForRow, ...
                        self.adjustTestImages);
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
    
    methods(Access = private)
        function path = appendSlash(~, path)
            if ~strendswith(path, '/')
                path = [path, '/'];
            end
        end
    end
end
