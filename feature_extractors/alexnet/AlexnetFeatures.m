classdef AlexnetFeatures < FeatureExtractor
    % Extract Alexnet features
    
    properties
        netParams
        imagesMean
    end
    
    methods
        function obj = AlexnetFeatures(netParams)
            dir = fileparts(mfilename('fullpath'));
            if ~exist('netParams', 'var') || isempty(netParams)
                netParams = load([dir, '/ressources/alexnetParams.mat']);
                % obtained from https://drive.google.com/file/d/0B-VdpVMYRh-pQWV1RWt5NHNQNnc/view
            end
            
            obj.netParams = netParams;
            imagesMeanData = load([dir '/ressources/ilsvrc_2012_mean.mat']);
            obj.imagesMean = imagesMeanData.mean_data;
        end
        
        function features = extractFeatures(self, images, ~, ~)
            for img = 1:length(images)
                preparedImage = prepareGrayscaleImage(images{img}, self.imagesMean);
                imageFeatures = self.getImageFeatures(preparedImage);
                if img == 1
                    features = zeros([length(images), size(imageFeatures)]);
                end
                features(img, :) = imageFeatures(:);
            end
        end
    end
    
    methods(Abstract)
        getImageFeatures(self, image, runType)
    end
end
