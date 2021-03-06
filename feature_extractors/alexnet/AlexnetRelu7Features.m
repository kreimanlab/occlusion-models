classdef AlexnetRelu7Features < AlexnetFeatures
    properties
        lowerExtractor
    end
    
    methods
        function obj = AlexnetRelu7Features(netParams, lowerExtractor)
            if ~exist('netParams', 'var')
                netParams = [];
            end
            obj = obj@AlexnetFeatures(netParams);
            if ~exist('lowerExtractor', 'var')
                lowerExtractor = AlexnetFc7Features(obj.netParams);
            end
            obj.lowerExtractor = lowerExtractor;
        end
        
        function name = getName(~)
            name = 'alexnet-relu7';
        end
        
        function dropout7 = getImageFeatures(self, image)
            % pass image through network
            fc7 = self.lowerExtractor.getImageFeatures(image);
            relu7 = relu(fc7);
            dropout7 = dropout(relu7);
        end
    end
end
