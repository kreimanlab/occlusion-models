classdef ImageProvider < FeatureExtractor
    % Provides images for a given dataset row
    
    properties
        data
        consumer
        images
        averageSpectra
    end
    
    methods
        function self = ImageProvider(consumer, data, images)
            self.consumer = consumer;
            self.data = data;
            self.images = images;
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
            images = self.images(self.data.pres(dataSelection));
            if runType == RunType.Test
                images = occlude(images, dataSelection, self.data);
            end
        end
    end
end
