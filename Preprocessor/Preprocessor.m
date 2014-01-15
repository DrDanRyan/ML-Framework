classdef Preprocessor < matlab.mixin.Copyable
   % This defines the Preprocessor interface
   
   methods (Abstract)
      x = feed_forward(obj, x)
      
      % If the preprocessor uses internal parameters it should also respond
      % to:
      % gather(obj)
      % push_to_GPU(obj)
      % init_params(obj)
   end
   
end

