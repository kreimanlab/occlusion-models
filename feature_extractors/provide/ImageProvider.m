classdef ImageProvider < FeatureExtractor
    % Provides images for a given dataset row
    
    properties
        consumer
        images
        objectForRow
        numsBubbles
        bubbleCenters
        bubbleSigmas
        averageSpectra
    end
    
    methods
        function self = ImageProvider(consumer, images, ...
                objectForRow, numsBubbles, bubbleCenters, bubbleSigmas)
            self.consumer = consumer;
            self.images = images;
            self.objectForRow = objectForRow;
            self.numsBubbles = numsBubbles;
            self.bubbleCenters = bubbleCenters;
            self.bubbleSigmas = bubbleSigmas;
        end
        
        function name = getName(self)
            name = self.consumer.getName();
        end
        
        function features = extractFeatures(self, dataSelection, ...
                runType, labels)
            imgs = self.getImages(dataSelection, runType);
            features = self.consumer.extractFeatures(imgs, runType, labels);
        end 
    end
    
    methods(Access = protected)
        function images = getImages(self, dataSelection, runType)
            images = self.images(self.objectForRow(dataSelection));
            if runType == RunType.Test
                nums = self.numsBubbles(dataSelection);
                centers = self.bubbleCenters(dataSelection, :);
                sigmas = self.bubbleSigmas(dataSelection, :);
                images = occlude(images, nums, centers, sigmas);
            end
        end
    end
end
