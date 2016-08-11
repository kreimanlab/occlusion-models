function varargin = replaceVararg(varargin, key, newValue)
i = find(strcmp(varargin, key));
varargin{i + 1} = newValue;
end
