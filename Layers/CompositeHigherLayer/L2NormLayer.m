classdef L2NormLayer < CompositeHigherLayer & matlab.mixin.Copyable
   % Treats all dimensions of input beyond second dimension as a single
   % dimension. Computes L2 norm over this last dimension and stores value in
   % first dimension. 
   
   properties
      dydx
   end
   
   methods
      function y = feed_forward(obj, x, isSave)
         D = ndims(x);
         y = x.*x;
         for i = D:-1:3
            y = sum(y, i);
         end
         y = sqrt(y);
         
         if nargin == 3 && isSave
            obj.dydx = bsxfun(@rdivide, x, y);
         end
      end
      
      function dLdx = backprop(obj, dLdy)
         dLdx = bsxfun(@times, dLdy, obj.dydx);
         obj.dydx = [];
      end
   end
   
end

