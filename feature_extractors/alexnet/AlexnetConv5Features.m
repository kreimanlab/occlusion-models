classdef AlexnetConv5Features < AlexnetFeatures    
    properties
        lowerExtractor
    end
    
    methods
        function obj = AlexnetConv5Features(netParams)
            if ~exist('netParams', 'var')
                netParams = [];
            end
            obj = obj@AlexnetFeatures(netParams);
            obj.lowerExtractor = AlexnetConv4Features(obj.netParams);
        end
        
        function name = getName(~)
            name = 'alexnet-conv4';
        end
        
        function conv5 = getImageFeatures(self, image)
            % Preparation
            conv5Kernels = self.netParams.weights(5).weights{1};
            conv5Bias = self.netParams.weights(5).weights{2};
            % pass image through network
            conv4 = self.lowerExtractor.getImageFeatures(image);
            relu4 = relu(conv4);
            conv5 = conv(relu4, conv5Kernels, conv5Bias, 3, 1, 1, 2);
        end
    end
end
