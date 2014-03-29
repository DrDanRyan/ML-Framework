classdef L1PenaltyLayer < matlab.mixin.Copyable
   
   properties
      C
      penalty
   end
   
   methods
      function obj = L1PenaltyLayer(C)
         obj.C = C;
      end
      
      function x = feed_forward(obj, x, isSave)         
         if nargin == 3 && isSave
            obj.penalty = obj.C*sign(x);
         end
      end
      
      function dLdx = backprop(obj, dLdy)
         dLdx = dLdy + obj.penalty;
         obj.penalty = [];
      end
   end
   
end