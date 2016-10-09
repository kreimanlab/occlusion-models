function runCreateHMAXFeatures(dataset_name,first_im_num,last_im_num)

if strcmp(dataset_name,'KLAB16')
    img_list={};
    for i=1:16
        img_list{end+1,1}=['/home/bill/Data/Occluded_Datasets/' dataset_name '/images/im_' num2str(i) '.tif'];
    end
    img_tags=1:16;
elseif strcmp(dataset_name,'KLAB25')
    img_list={};
    for i=1:25
        img_list{end+1,1}=['/home/bill/Data/Occluded_Datasets/' dataset_name '/images/im_' num2str(i) '.tif'];
    end
    img_tags=1:25;
elseif strcmp(dataset_name,'KLAB325v2')
    img_list={};
    for i=1:325
        img_list{end+1,1}=['/home/bill/Data/Occluded_Datasets/' dataset_name '/images/im_' num2str(i) '.tif'];
    end
    img_tags=1:325;
elseif strcmp(dataset_name,'KLAB325')
    img_list={};
    cats={'Animals','Chairs','Faces','Fruits','Vehicles'};
    base_dir=['/home/bill/Data/Occluded_Datasets/' dataset_name '/images/'];
    for c=1:length(cats)
        for i=1:65
            img_list{end+1,1}=[base_dir cats{c} '-' num2str(i) '.tif'];
        end
    end
    img_tags=1:325;
else
    f=['/home/bill/Data/Occluded_Datasets/' dataset_name '/files/' dataset_name '_im_list.txt'];
    img_list=splitlines(f);
    img_list=img_list(first_im_num:last_im_num);
    img_tags=zeros(length(img_list),1);
    for i=1:length(img_list)
        idx0=strfind(img_list{i},'_');
        idx0=idx0(end);
        idx1=strfind(img_list{i},'.');
        idx1=idx1(end);
        img_tags(i)=str2double(img_list{i}(idx0+1:idx1-1));
    end
end

ims=cell(length(img_list),1);
for i=1:length(img_list)
    im=imread(img_list{i});
    im=im2double(im);
    if size(im,3)>1
        im=rgb2gray(im);
    end
    ims{i}=im;
end

save_file=['/home/bill/Data/Occluded_Datasets/' dataset_name '/features/hmax_ims_' num2str(img_tags(1)) '-' num2str(img_tags(end)) '.mat'];
createHMAXFeatures(ims, img_tags, save_file)


end