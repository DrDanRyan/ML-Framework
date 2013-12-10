classdef ReluNoParamsLayer < NoParamsLayer
   
   methods
      function y = feed_forward(obj, x, isSave)
         y = max(0, x);
         
         if nargin == 3 && isSave
            obj.Dy = x > 0; % store as logical to reduce memory usage
         end
      end
      
      function dLdx = backprop(obj, dLdy)
         if isa(dLdy, 'gpuArray')
            dLdx = single(obj.Dy).*dLdy;
         else
            dLdx = obj.Dy.*dLdy;
         end
         obj.Dy = [];
      end
   end
end

