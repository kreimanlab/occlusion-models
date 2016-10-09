function createHMAXFeatures(ims, img_tags, save_file)


% ims should be 256 by 256 grayscale double images
addpath('/home/bill/Dropbox/WLotter/standardmodel')

patchSizes = [4 8 12 16];               % other sizes might be better, maybe not all sizes are required
numPatchSizes = length(patchSizes);


numPatchesPerSize = 250;                % more will give better results, but will take more time to compute

load('cPatches.mat');

cI=ims;
%----Settings for Testing --------%
rot = [90 -45 0 45];
c1ScaleSS = [1:2:18];
RF_siz    = [7:2:39];
c1SpaceSS = [8:2:22];
minFS     = 7;
maxFS     = 39;
div = [4:-.05:3.2];
Div       = div;
%--- END Settings for Testing --------%

% creates the gabor filters use to extract the S1 layer
disp('Initializing gabor filters -- full set...');
[fSiz,filters,c1OL,numSimpleFilters] = init_gabor(rot, RF_siz, Div);
disp('done');

C2res = extractC2forcell(filters,fSiz,c1SpaceSS,c1ScaleSS,c1OL,cPatches,cI,numPatchSizes);


features=C2res';

save(save_file,'features','img_tags');
    

end


