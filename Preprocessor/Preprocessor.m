classdef Preprocessor
   % This defines the Preprocessor interface
   
   methods (Abstract)
      data = transform(obj, data)
   end
   
end

