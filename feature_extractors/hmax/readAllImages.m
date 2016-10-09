function cI = readAllImages(train_set,test_set,maximagesperdir);
%Reads all training and testing images into a cell of length 4
%cI{1} = train_set.pos,
%cI{2} = train_set.neg,
%cI{3} = test_set.pos,
%cI{4} = test_set.neg,
if nargin<3
  maximagesperdir = inf;
end

dnames = {train_set.pos,train_set.neg,test_set.pos,test_set.neg};

fprintf('Reading images...');    
cI = cell(4,1);
for i = 1:4,
  c{i} = dir(dnames{i});
  if length(c{i})>0,
    if c{i}(1).name == '.',
      c{i} = c{i}(3:end);
    end
  end
  if length(c{i})>maximagesperdir,
    c{i} = c{i}(1:maximagesperdir);
  end
  cI{i} = cell(length(c{i}),1);
  for j = 1:length(c{i}),
    cI{i}{j} = double(imread([dnames{i} '/' c{i}(j).name]))./255;
  end
end

fprintf('done.\n');
