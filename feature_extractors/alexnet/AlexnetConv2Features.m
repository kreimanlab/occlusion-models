classdef AlexnetConv2Features < AlexnetFeatures    
    properties
        lowerExtractor
    end
    
    methods
        function obj = AlexnetConv2Features(netParams)
            if ~exist('netParams', 'var')
                netParams = [];
            end
            obj = obj@AlexnetFeatures(netParams);
            obj.lowerExtractor = AlexnetConv1Features(obj.netParams);
        end
        
        function name = getName(~)
            name = 'alexnet-conv2';
        end
        
        function conv2 = getImageFeatures(self, image)
            % Preparation
            conv2Kernels = self.netParams.weights(2).weights{1};
            conv2Bias = self.netParams.weights(2).weights{2};
            % pass image through network
            conv1 = self.lowerExtractor.getImageFeatures(image);
            relu1 = relu(conv1);
            pool1 = maxpool(relu1, 3, 2);
            lrn1 = lrn(pool1, 5, .0001, 0.75, 1);
            conv2 = conv(lrn1, conv2Kernels, conv2Bias, 5, 1, 2, 2);
        end
    end
end
