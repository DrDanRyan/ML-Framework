classdef L2NormalizeLayer < matlab.mixin.Copyable
   
   properties
      xNorm
      y
   end
   
   methods
      function y = feed_forward(obj, x, isSave)
         xNorm = sqrt(sum(x.*x)); %#ok<*PROP>
         y = bsxfun(@rdivide, x, xNorm);
         
         if nargin == 3 && isSave
            obj.xNorm = xNorm;
            obj.y = y;
         end
      end
      
      function dLdx = backprop(obj, dLdy)
         dLdx = bsxfun(@rdivide, dLdy, obj.xNorm) - ...
                bsxfun(@times, obj.y, sum(obj.y.*dLdy)./obj.xNorm);
         obj.xNorm = [];
         obj.y = [];
      end
   end
   
end

