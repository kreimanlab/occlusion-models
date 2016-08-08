classdef AlexnetRelu3Features < AlexnetReluFeatures
    methods
        function obj = AlexnetRelu3Features(netParams)
            if ~exist('netParams', 'var')
                netParams = [];
            end
            obj = obj@AlexnetReluFeatures(netParams, ...
                @AlexnetConv3Features);
        end
        
        function name = getName(~)
            name = 'alexnet-relu3';
        end
    end
end
