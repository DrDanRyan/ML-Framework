classdef StandardOutputLayer < OutputLayer & StandardLayer
   
   methods
      function obj = StandardOutputLayer(inputSize, outputSize, varargin)
         obj = obj@StandardLayer(inputSize, outputSize, varargin{:});
      end
      
      function [grad, dLdx, y] = backprop(obj, x, t)
         [dLdz, y] = obj.compute_dLdz(x, t);
         dLdx = obj.params{1}'*dLdz;
         grad = obj.grad_from_dLdz(x, dLdz);
      end
   end
   
   methods (Abstract)
      [dLdz, y] = compute_dLdz(obj, x, t)
   end
   
end

