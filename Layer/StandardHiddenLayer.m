classdef StandardHiddenLayer < HiddenLayer & StandardLayer
   
   methods
      function obj = StandardHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardLayer(inputSize, outputSize, varargin{:});
      end
      
      function [grad, dLdx] = backprop(obj, x, y, dLdy, isAveraged)
         if nargin < 5
            isAveraged = true;
         end
         dLdz = dLdy.*obj.dydz(y);
         dLdx = obj.params{1}'*dLdz;
         grad = obj.grad_from_dLdz(x, dLdz, isAveraged);
      end
   end
   
   methods (Abstract)
      value = dydz(~, y)
   end
   
end

