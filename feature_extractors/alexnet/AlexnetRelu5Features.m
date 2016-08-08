classdef AlexnetRelu5Features < AlexnetReluFeatures
    methods
        function obj = AlexnetRelu5Features(netParams)
            if ~exist('netParams', 'var')
                netParams = [];
            end
            obj = obj@AlexnetReluFeatures(netParams, ...
                @AlexnetConv5Features);
        end
        
        function name = getName(~)
            name = 'alexnet-relu5';
        end
    end
end
