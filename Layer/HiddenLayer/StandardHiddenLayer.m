classdef StandardHiddenLayer < HiddenLayer & StandardLayer
   % Easiest point to subclass for a simple HiddenLayer. Only need to
   % provide: feedforward and compute_Dy in subclass
   
   methods
      function obj = StandardHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardLayer(inputSize, outputSize, varargin{:});
      end
      
      function [grad, dLdx, Dy] = backprop(obj, x, y, dLdy)
         Dy = obj.compute_Dy(x, y);
         dLdz = dLdy.*Dy;
         dLdx = obj.params{1}'*dLdz;
         grad = obj.grad_from_dLdz(x, dLdz);
      end
   end
   
   methods (Abstract)
      Dy = compute_Dy(x, y)
   end
   
end

