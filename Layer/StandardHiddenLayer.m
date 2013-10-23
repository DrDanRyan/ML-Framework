classdef StandardHiddenLayer < HiddenLayer & StandardLayer
   
   methods
      function obj = StandardHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardLayer(inputSize, outputSize, varargin{:});
      end
      
      function [grad, dLdx, dydz] = backprop(obj, x, y, dLdy)
         dydz = obj.dydz([], y);
         dLdz = dLdy.*dydz;
         dLdx = obj.params{1}'*dLdz;
         grad = obj.grad_from_dLdz(x, dLdz);
      end
   end
   
end

