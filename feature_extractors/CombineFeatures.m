classdef CombineFeatures < FeatureExtractor
    % Combines features from two or more feature extractors.
    
    properties
        combineFnc
        featureInputs
    end
    
    methods
        function obj = CombineFeatures(combineFnc, varargin)
            obj.combineFnc = combineFnc;
            obj.featureInputs = varargin;
        end
        
        function name = getName(self)
            name = func2str(self.combineFnc);
            for i = 1:numel(self.featureInputs)
                name = [name, '_', self.featureInputs{i}.getName()];
            end
        end
        
        function features = extractFeatures(self, rows, ...
                runType, labels)
            for i = 1:numel(self.featureInputs)
                inputFeatures = self.featureInputs{i}.extractFeatures(...
                    rows, runType, labels);
                if ~exist('features', 'var')
                    features = inputFeatures;
                else
                    features = self.combineFnc(features, inputFeatures);
                end
            end
        end
        
        function net = trainNet(self, features)
            T = features';
            net = newhop(T);
            self.netTrained = true;
        end
    end
end
