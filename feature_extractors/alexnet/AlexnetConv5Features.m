classdef AlexnetConv5Features < AlexnetFeatures    
    properties
        lowerExtractor
    end
    
    methods
        function obj = AlexnetConv5Features(netParams, lowerExtractor)
            if ~exist('netParams', 'var')
                netParams = [];
            end
            obj = obj@AlexnetFeatures(netParams);
            if ~exist('lowerExtractor', 'var')
                lowerExtractor = AlexnetRelu4Features(obj.netParams);
            end
            obj.lowerExtractor = lowerExtractor;
        end
        
        function name = getName(~)
            name = 'alexnet-conv5';
        end
        
        function conv5 = getImageFeatures(self, image)
            % Preparation
            conv5Kernels = self.netParams.weights(5).weights{1};
            conv5Bias = self.netParams.weights(5).weights{2};
            % pass image through network
            relu4 = self.lowerExtractor.getImageFeatures(image);
            conv5 = conv(relu4, conv5Kernels, conv5Bias, 3, 1, 1, 2);
        end
    end
end
