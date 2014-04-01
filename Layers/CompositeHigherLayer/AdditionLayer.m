classdef AdditionLayer < CompositeHigherLayer & matlab.mixin.Copyable
   % Adds a constant value to the input. Useful to fudge nonnegative values
   % to be strictly positive (which is particularly useful for sparse
   % filtering)
   
   properties
      eps
   end
   
   methods
      function obj = AdditionLayer(eps)
         if nargin == 0
            obj.eps = 1e-6;
         else
            obj.eps = eps;
         end
      end
      
      function y = feed_forward(obj, x, ~)
         y = x + obj.eps;
      end
      
      function dLdx = backprop(obj, dLdy)
         dLdx = dLdy;
      end
      
   end
end

