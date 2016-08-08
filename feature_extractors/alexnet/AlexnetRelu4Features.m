classdef AlexnetRelu4Features < AlexnetReluFeatures
    methods
        function obj = AlexnetRelu4Features(netParams)
            if ~exist('netParams', 'var')
                netParams = [];
            end
            obj = obj@AlexnetReluFeatures(netParams, ...
                @AlexnetConv4Features);
        end
        
        function name = getName(~)
            name = 'alexnet-relu4';
        end
    end
end
