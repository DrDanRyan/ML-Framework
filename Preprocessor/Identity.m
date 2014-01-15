classdef Identity < Preprocessor
   % transform does nothing... useful to plug into a framework where
   % preprocessors are defined but you want to deal with the original data
   
   methods
      function data = transform(obj, data)
         % pass
      end
      
      function gather(obj)
         % pass
      end
      
      function push_to_GPU(obj)
         % pass
      end
   end
   
end

