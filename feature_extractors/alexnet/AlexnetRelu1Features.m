classdef AlexnetRelu1Features < AlexnetReluFeatures
    methods
        function obj = AlexnetRelu1Features(netParams)
            if ~exist('netParams', 'var')
                netParams = [];
            end
            obj = obj@AlexnetReluFeatures(netParams, ...
                @AlexnetConv1Features);
        end
        
        function name = getName(~)
            name = 'alexnet-relu1';
        end
    end
end
