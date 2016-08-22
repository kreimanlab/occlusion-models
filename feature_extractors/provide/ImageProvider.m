classdef ImageProvider < FeatureExtractor
    % Provides images for a given dataset row
    
    properties
        consumer
        images
        objectForRow
        modifyTestImages
    end
    
    methods
        function self = ImageProvider(consumer, images, ...
                objectForRow, modifyTestImages)
            self.consumer = consumer;
            self.images = images;
            self.objectForRow = objectForRow;
            self.modifyTestImages = modifyTestImages;
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
                images = self.modifyTestImages(images, dataSelection);
            end
        end
    end
end
