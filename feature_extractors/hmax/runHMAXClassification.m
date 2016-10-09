function runHMAXClassification(images, train_idx, test_idx, images_for_patches, save_dir)


% ims should be 256 by 256 grayscale double images


patchSizes = [4 8 12 16];               % other sizes might be better, maybe not all sizes are required
numPatchSizes = length(patchSizes);

addpath('C:\Users\WLotter\Dropbox\Classes\Computer_vision\Final_Project\src')
[all_images,all_labels]=read_test_images;



numPatchesPerSize = 250;                % more will give better results, but will take more time to compute
cPatches = extractRandC1Patches(images_for_patches, numPatchSizes, numPatchesPerSize, patchSizes);
if ~exist(save_dir,'dir')
    mkdir(save_dir);
end
save([save_dir 'cPatches.mat'],'cPatches');


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

%The actual C2 features are computed below for each one of the training/testing directories
%tic
%for i = 1:4,
C2res = extractC2forcell(filters,fSiz,c1SpaceSS,c1ScaleSS,c1OL,cPatches,cI,numPatchSizes);
%  toc
%end

%Simple classification code
train_stop=floor(train_prop*length(all_labels));

cI=all_images(1:train_stop);
cI_test=all_images(1+train_stop:end);
train_labels=all_labels(1:train_stop);
test_labels=all_labels(1+train_stop:end);

% XTrain=[];
% for i=1:train_stop
%     XTrain=[XTrain C2res{i}];
% end
% XTest=[];
% for i=train_stop+1:length(all_labels)
%     XTest=[XTest C2res{i}];
% end
XTrain=C2res(:,1:train_stop);
XTest=C2res(:,1+train_stop:end);

ytrain=all_labels(1:train_stop);
ytest=all_labels(1+train_stop:end);
%XTrain = [C2res{1} C2res{2}]; %training examples as columns
%XTest =  [C2res{3},C2res{4}]; %the labels of the training set
%ytrain = [ones(size(C2res{1},2),1);-ones(size(C2res{2},2),1)];%testing examples as columns
%ytest = [ones(size(C2res{3},2),1);-ones(size(C2res{4},2),1)]; %the true labels of the test set
gammas=logspace(-4,4,8);
costs=logspace(-2,8,10);

model=fit_svm(gammas,costs,XTrain',ytrain,1,[],[],0);
[ry,this_acc,~]=svmpredict(ytest,XTest',model);

successrate = mean(ytest==ry) %a simple classification score

end


