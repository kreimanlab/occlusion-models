classdef AlexnetRelu2Features < AlexnetReluFeatures
    methods
        function obj = AlexnetRelu2Features(netParams)
            if ~exist('netParams', 'var')
                netParams = [];
            end
            obj = obj@AlexnetReluFeatures(netParams, ...
                @AlexnetConv2Features);
        end
        
        function name = getName(~)
            name = 'alexnet-relu2';
        end
    end
end
