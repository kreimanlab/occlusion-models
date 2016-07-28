classdef RnnFeatures < FeatureExtractor
    %RNNFEATURES Feature extractor for a RNN layer on top of other features
    % output_{t+1} = activation(previousFeatures + W .* output_t)
    
    properties
        timesteps
        featuresInput
        name
        weights
        activation = @(x) max(0, x)
    end
    
    methods
        function obj = RnnFeatures(timesteps, featuresInput, name)
            obj.timesteps = timesteps;
            obj.featuresInput = featuresInput;
            if ~exist('name', 'var')
                name = 'RNN_features_fc7_noRelu_t';
            end
            obj.name = name;
        end
        
        function name = getName(self)
            name = [self.name, num2str(self.timesteps)];
        end
        
        function previousFeatures = extractFeatures(self, rows, runType, labels)
            previousFeatures = self.featuresInput.extractFeatures(rows, runType, labels);
            if isempty(self.weights)
                self.weights = self.createWeights(size(previousFeatures, 2));
            end
            W = repmat(self.weights, size(previousFeatures, 1), 1);
            features = previousFeatures;
            for t = 1:self.timesteps
                features = self.activation(previousFeatures + W .* features);
            end
        end
        
        function weights = createWeights(~, ~)
            error('Not implemented');
        end
    end
end
