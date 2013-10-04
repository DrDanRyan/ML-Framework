classdef StandardOutputLayer < OutputLayer & StandardLayer
   
   % abstract property "nonlinearity" must also be implemented to subclass
   % this class
   
   methods
      function obj = StandardOutputLayer(inputSize, outputSize, varargin)
         obj = obj@StandardLayer(inputSize, outputSize, varargin{:});
      end
      
      function [grad, dLdx, y] = backprop(obj, x, t, isAveraged)
         if nargin < 4
            isAveraged = true;
         end
         [dLdz, y] = obj.dLdz(x, t); 
         dLdx = obj.params{1}'*dLdz;
         grad = obj.grad_from_dLdz(x, dLdz, isAveraged);
      end
   end
   
   methods (Abstract)
      [dLdz, y] = dLdz(obj, x, t)
   end
   
end

