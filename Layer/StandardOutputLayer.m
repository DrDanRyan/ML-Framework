classdef StandardOutputLayer < OutputLayer & StandardLayer
   
   % abstract property "nonlinearity" must also be implemented to subclass
   % this class
   
   methods
      function obj = StandardOutputLayer(inputSize, outputSize, varargin)
         obj = obj@StandardLayer(inputSize, outputSize, varargin{:});
      end
      
      function [grad, dLdx, y] = backprop(obj, x, t, isAveraged, dLdz)
         if nargin < 4
            isAveraged = true;
         end
         
         if nargin < 5 % Useful to supply own dLdz if it needs to be modified by a mask
            [dLdz, y] = obj.dLdz(x, t); 
         end
         
         dLdx = obj.params{1}'*dLdz;
         grad = obj.grad_from_dLdz(x, dLdz, isAveraged);
      end
   end
   
   methods (Abstract)
      [dLdz, y] = dLdz(obj, x, t)
   end
   
end

