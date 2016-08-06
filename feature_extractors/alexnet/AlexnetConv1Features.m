classdef AlexnetConv1Features < AlexnetFeatures    
    methods
        function obj = AlexnetConv1Features(netParams)
            if ~exist('netParams', 'var')
                netParams = [];
            end
            obj = obj@AlexnetFeatures(netParams);
        end
        
        function name = getName(~)
            name = 'alexnet-conv1';
        end
        
        function conv1 = getImageFeatures(self, image)
            % Preparation
            conv1Kernels = self.netParams.weights(1).weights{1};
            conv1Bias = self.netParams.weights(1).weights{2};
            % pass image through network
            conv1 = conv(image, conv1Kernels, conv1Bias, 11, 4, 0, 1);
        end
    end
end

