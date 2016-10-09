function createHMAXPatches

addpath('/home/bill/Dropbox/WLotter/standardmodel')

im_dir='/home/bill/Data/GallantSecondDataset/Gray_1000/';

list=dir(im_dir);
p=randperm(length(list));
list=list(p);

n_ims=250;

ims={};

c=1;
while length(ims)<n_ims
    if ~isempty(strfind(list(c).name,'.png'))
        im_file=[im_dir list(c).name];
        im=imread(im_file);
        im=im2double(im);
        im=imresize(im,256/size(im,1));
        ims{end+1,1}=im;
    end
    c=c+1;
end


patchSizes = [4 8 12 16];               % other sizes might be better, maybe not all sizes are required
numPatchSizes = length(patchSizes);

numPatchesPerSize = 250;                % more will give better results, but will take more time to compute
cPatches = extractRandC1Patches(ims, numPatchSizes, numPatchesPerSize, patchSizes);
save(['cPatches.mat'],'cPatches');


end