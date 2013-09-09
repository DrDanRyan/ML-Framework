classdef Preprocessor
   % This defines the Preprocessor interface
   
   methods (Abstract)
      data = transform(obj, data)
      gather(obj)
      push_to_GPU(obj)
   end
   
end

