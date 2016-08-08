classdef AlexnetConv3Features < AlexnetFeatures    
    properties
        lowerExtractor
    end
    
    methods
        function obj = AlexnetConv3Features(netParams, lowerExtractor)
            if ~exist('netParams', 'var')
                netParams = [];
            end
            obj = obj@AlexnetFeatures(netParams);
            if ~exist('lowerExtractor', 'var')
                lowerExtractor = AlexnetRelu2Features(obj.netParams);
            end
            obj.lowerExtractor = lowerExtractor;
        end
        
        function name = getName(~)
            name = 'alexnet-conv3';
        end
        
        function conv3 = getImageFeatures(self, image)
            % Preparation
            conv3Kernels = self.netParams.weights(3).weights{1};
            conv3Bias = self.netParams.weights(3).weights{2};
            % pass image through network
            relu2 = self.lowerExtractor.getImageFeatures(image);
            pool2 = maxpool(relu2, 3, 2);
            norm2 = lrn(pool2, 5, .0001, 0.75, 1);
            conv3 = conv(norm2, conv3Kernels, conv3Bias, 3, 1, 1, 1);
        end
    end
end
