function varargin = replaceOrAddVararg(varargin, key, newValue)
i = find(strcmp(varargin, key));
if ~isempty(i)
    varargin{i + 1} = newValue;
else
    varargin{end + 1} = key;
    varargin{end + 1} = newValue;
end
end
