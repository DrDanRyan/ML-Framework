function flag = check_nan(varargin)
flag = false;
for i = 1:length(varargin)
   if any(isnan(varargin{i}(:)))
      flag = true;
   end
end

