classdef AlexnetFc8Features < AlexnetFeatures
    % Extract Alexnet fc8 features
    
    properties
        lowerExtractor
    end
    
    methods
        function obj = AlexnetFc8Features(netParams, lowerExtractor)
            if ~exist('netParams', 'var')
                netParams = [];
            end
            obj = obj@AlexnetFeatures(netParams);
            if ~exist('lowerExtractor', 'var')
                lowerExtractor = AlexnetRelu7Features(obj.netParams);
            end
            obj.lowerExtractor = lowerExtractor;
        end
        
        function name = getName(~)
            name = 'alexnet-fc8';
        end
        
        function fc8 = getImageFeatures(self, image)
            %% Preparation
            fc8Weights = self.netParams.weights(8).weights{1};
            fc8Bias = self.netParams.weights(8).weights{2};

            %% pass image through network
            relu7 = self.lowerExtractor.getImageFeatures(image);
            dropout7 = dropout(relu7);
            fc8 = fc(dropout7, fc8Weights, fc8Bias);
        end
    end
end
