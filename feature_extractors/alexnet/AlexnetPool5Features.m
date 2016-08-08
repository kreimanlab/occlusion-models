classdef AlexnetPool5Features < AlexnetFeatures    
    properties
        lowerExtractor
    end
    
    methods
        function obj = AlexnetPool5Features(netParams, lowerExtractor)
            if ~exist('netParams', 'var')
                netParams = [];
            end
            obj = obj@AlexnetFeatures(netParams);
            if ~exist('lowerExtractor', 'var')
                lowerExtractor = AlexnetRelu5Features(obj.netParams);
            end
            obj.lowerExtractor = lowerExtractor;
        end
        
        function name = getName(~)
            name = 'alexnet-pool5';
        end
        
        function pool5_2d = getImageFeatures(self, image)
            % pass image through network
            relu5 = self.lowerExtractor.getImageFeatures(image);
            pool5 = maxpool(relu5, 3, 2);
            pool5_2d = reshape(pool5, [9216, 1]); % flatten data
        end
    end
end
