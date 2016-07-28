classdef Classifier < handle
    %Interface for classifiers
    
    properties
        featureExtractor
    end
    
    properties (Abstract)
        name
    end
    
    methods
        function obj = Classifier(featureExtractor)
            obj.featureExtractor = featureExtractor;
        end
        
        function name = getName(self)
            name = [self.featureExtractor.getName() '-' self.name];
        end
        
        function train(self, X, Y)
            features = self.featureExtractor.extractFeatures(X, ...
                RunType.Train, Y);
            self.fit(features, Y);
        end
        
        function labels = predict(self, X)
            features = self.featureExtractor.extractFeatures(X, ...
                RunType.Test, []);
            labels = self.classify(features);
        end
    end
    
    methods (Abstract)
        fit(self, features, labels)
        classify(self, features)
    end
end

