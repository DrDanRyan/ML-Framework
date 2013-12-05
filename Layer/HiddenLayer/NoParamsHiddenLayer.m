classdef NoParamsHiddenLayer < HiddenLayer & NoParamsLayer
   
   methods
      function [grad, dLdx, Dy] = backprop(obj, x, y, ~, dLdy)
         grad = [];
         Dy = obj.compute_Dy(x, y);
         dLdx = Dy.*dLdy;
      end
   end
   
   methods (Abstract)
      Dy = compute_Dy(obj, x, y)
   end
end

