classdef ReluNoParamsLayer < NoParamsLayer
   
   methods
      function y = feed_forward(obj, x, isSave)
         y = max(0, x);
         
         if nargin == 3 && isSave
            obj.Dy = x > 0;
            if isa(x, 'gpuArray')
               obj.Dy = single(obj.Dy);
            end
         end
      end   
   end
end

