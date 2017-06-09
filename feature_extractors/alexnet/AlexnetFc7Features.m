classdef AlexnetFc7Features < AlexnetFeatures
    properties
        lowerExtractor
    end
    
    methods
        function obj = AlexnetFc7Features(netParams, lowerExtractor)
            if ~exist('netParams', 'var')
                netParams = [];
            end
            obj = obj@AlexnetFeatures(netParams);
            if ~exist('lowerExtractor', 'var')
                lowerExtractor = AlexnetRelu6Features(obj.netParams);
            end
            obj.lowerExtractor = lowerExtractor;
        end
        
        function name = getName(~)
            name = 'alexnet-fc7';
        end
        
        function fc7 = getImageFeatures(self, image)
            % Preparation
            fc6Weights=self.netParams.weights(6).weights{1};
            fc6Bias=self.netParams.weights(6).weights{2};
            fc7Weights=self.netParams.weights(7).weights{1};
            fc7Bias=self.netParams.weights(7).weights{2};
            % pass image through network
            pool5_2d = self.lowerExtractor.getImageFeatures(image);
            fc6 = fc(pool5_2d, fc6Weights, fc6Bias);
            relu6 = relu(fc6);
            dropout6 = dropout(relu6);
            fc7 = fc(dropout6, fc7Weights, fc7Bias);
        end
    end
end
