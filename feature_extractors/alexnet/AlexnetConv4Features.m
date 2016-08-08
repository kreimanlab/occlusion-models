classdef AlexnetConv4Features < AlexnetFeatures    
    properties
        lowerExtractor
    end
    
    methods
        function obj = AlexnetConv4Features(netParams, lowerExtractor)
            if ~exist('netParams', 'var')
                netParams = [];
            end
            obj = obj@AlexnetFeatures(netParams);
            if ~exist('lowerExtractor', 'var')
                lowerExtractor = AlexnetRelu3Features(obj.netParams);
            end
            obj.lowerExtractor = lowerExtractor;
        end
        
        function name = getName(~)
            name = 'alexnet-conv4';
        end
        
        function conv4 = getImageFeatures(self, image)
            % Preparation
            conv4Kernels = self.netParams.weights(4).weights{1};
            conv4Bias = self.netParams.weights(4).weights{2};
            % pass image through network
            relu3 = self.lowerExtractor.getImageFeatures(image);
            conv4 = conv(relu3, conv4Kernels, conv4Bias, 3, 1, 1, 2);
        end
    end
end
