classdef LibsvmClassifier < Classifier
    %LIBSVMCLASSIFIER Libsvm One-vs-all SVM Classifier.
    
    properties (Constant)
        kernelNames = containers.Map([0, 1, 2, 3], ...
            {'linear', 'polynomial', 'radial', 'sigmoid'});
    end
    
    properties
        name
        model
        kernel % kernel type
        cost % the C in C-SVC
        numTrainedFeatures
    end
    
    methods
        function obj = LibsvmClassifier(featureExtractor, kernel, cost)
            obj@Classifier(featureExtractor);
            if ~exist('kernel', 'var')
                kernel = 0;
            end
            assert(isnumeric(kernel) || ischar(kernel), ...
                'kernel must either be the numeric type or the name');
            if ischar(kernel)
                keys = obj.kernelNames.keys();
                kernel = keys{strcmp(obj.kernelNames.values(), kernel)};
            end
            obj.kernel = kernel;
            if ~exist('cost', 'var')
                cost = 1;
            end
            obj.cost = cost;
            obj.name = ['libsvm_' obj.kernelNames(obj.kernel)];
        end
        
        function fit(self, X, Y)
            self.model = libsvmtrain(Y, X, ...
                ['-q -t ' num2str(self.kernel) ' -c ' num2str(self.cost)]);
            self.numTrainedFeatures = size(X, 2);
        end
        
        function Y = classify(self, X)
            if size(X, 2) ~= self.numTrainedFeatures
                error(['invalid number of features: ' ...
                    'expected ' num2str(self.numTrainedFeatures) ...
                    ', got ' num2str(size(X, 2)) ]);
            end
            Y = libsvmpredict(rand(size(X, 1), 1), ...
                X, self.model, '-q');
        end
    end
end
