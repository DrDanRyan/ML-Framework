function flag = check_nan(varargin)
% Utility function used to test if any input array contains a NaN value.
% Useful for debugging source of NaN values.

flag = false;
for i = 1:length(varargin)
   if any(isnan(varargin{i}(:)))
      flag = true;
   end
end

