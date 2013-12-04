classdef NoParamsLayer < handle
   % A mixin that provides an empty facade for parameter related layer
   % interface functions
   
   methods
      function init_params(~)
      end
      
      function increment_params(~, ~)
      end
      
      function push_to_GPU(~)
      end
      
      function gather(~)
      end
   end
end

