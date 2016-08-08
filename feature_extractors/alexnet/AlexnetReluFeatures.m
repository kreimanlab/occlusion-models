classdef AlexnetReluFeatures < AlexnetFeatures    
    properties
        lowerExtractor
    end
    
    methods
        function obj = AlexnetReluFeatures(netParams, lowerExtractorCtr)
            obj = obj@AlexnetFeatures(netParams);
            obj.lowerExtractor = lowerExtractorCtr(obj.netParams);
        end
        
        function reluOut = getImageFeatures(self, image)
            % pass image through network
            conv = self.lowerExtractor.getImageFeatures(image);
            reluOut = relu(conv);
        end
    end
end
