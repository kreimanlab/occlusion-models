classdef HmaxFeatures < FeatureExtractor
    % Extracts features using HMAX.
    
    properties (Access = private)
        numPatchSizes
        cPatches
        c1ScaleSS
        c1SpaceSS
        fSiz
        filters
        c1OL
    end
    
    methods
        function obj = HmaxFeatures(patchImages)
            %% create patches
            patchSizes = [4 8 12 16];
            obj.numPatchSizes = length(patchSizes);
            numPatchesPerSize = 250;
            if exist('patchImages', 'var')
                obj.cPatches = extractRandC1Patches(patchImages, ...
                    obj.numPatchSizes, numPatchesPerSize, patchSizes);
            else
                obj.cPatches = load('PatchesFromNaturalImages250per4sizes.mat');
                obj.cPatches = obj.cPatches.cPatches;
            end
            
            %% create the gabor filters use to extract the S1 layer
            rot = [90 -45 0 45];
            obj.c1ScaleSS = 1:2:18;
            RF_siz = 7:2:39;
            obj.c1SpaceSS = 8:2:22;
            div = 4:-.05:3.2;
            [obj.fSiz, obj.filters, obj.c1OL] = ...
                init_gabor(rot, RF_siz, div);
        end
        
        function name = getName(~)
            name = 'hmax';
        end
        
        function c2 = extractFeatures(obj, images, ~, ~)
            c2 = extractC2forcell(obj.filters, obj.fSiz, obj.c1SpaceSS, ...
                obj.c1ScaleSS, obj.c1OL, obj.cPatches, ...
                images, obj.numPatchSizes)';
            assert(length(images) == size(c2, 1));
        end
    end
end

