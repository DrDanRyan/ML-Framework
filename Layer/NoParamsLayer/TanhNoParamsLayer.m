classdef TanhNoParamsLayer < NoParamsLayer

   methods
      function y = feed_forward(obj, x, isSave)
         v = exp(-2*x);
         u = 2./(1 + v);
         y = u - 1; % robust tanh(x)
         
         if nargin == 3 && isSave
            obj.Dy = v.*u.*u;
         end
      end   
   end
end

