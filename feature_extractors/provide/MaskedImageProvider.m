classdef MaskedImageProvider < ImageProvider
    % Provides masked images for a given dataset row
    
    properties
        averageSpectra
    end
    
    methods
        function obj = MaskedImageProvider(consumer, data, images, ...
                averageSpectra)
            obj = obj@ImageProvider(consumer, data, images);
            obj.averageSpectra = averageSpectra;
        end
        
        function name = getName(self)
            name = [self.consumer.getName(), '-masked'];
        end
        
        function features = extractFeatures(self, dataSelection, ...
                runType, labels)
            imgs = self.getImages(dataSelection, runType);
            for i = 1:numel(imgs)
                imgs{i} = createPhaseScramble(...
                    size(imgs{i}), self.averageSpectra);
            end
            features = self.consumer.extractFeatures(imgs, runType, labels);
        end
    end
end
