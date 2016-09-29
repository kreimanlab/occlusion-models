function data = loadData(file, key)
data = load(file);
data = data.(key);
end
