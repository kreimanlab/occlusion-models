classdef AlexnetPool5Features < AlexnetFeatures    
    properties
        lowerExtractor
    end
    
    methods
        function obj = AlexnetPool5Features(netParams)
            if ~exist('netParams', 'var')
                netParams = [];
            end
            obj = obj@AlexnetFeatures(netParams);
            obj.lowerExtractor = AlexnetConv5Features(obj.netParams);
        end
        
        function name = getName(~)
            name = 'alexnet-pool5';
        end
        
        function pool5_2d = getImageFeatures(self, image)
            % pass image through network
            conv5 = self.lowerExtractor.getImageFeatures(image);
            relu5 = relu(conv5);
            pool5 = maxpool(relu5, 3, 2);
            pool5_2d = reshape(pool5, [9216, 1]); % flatten data
        end
    end
end
