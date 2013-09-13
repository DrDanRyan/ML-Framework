classdef StandardOutputLayer < OutputLayer & StandardLayer
   
   % abstract property "nonlinearity" must also be implemented to subclass
   % this class
   
   methods
      function obj = StandardOutputLayer(inputSize, outputSize, varargin)
         obj = obj@StandardLayer(inputSize, outputSize, varargin{:});
      end
      
      function [grad, dLdx, y] = backprop(obj, x, t)
         N = size(x, 2);
         [dLdz, y] = obj.dLdz(x, t); 
         dLdx = obj.params{1}'*dLdz;

         if obj.isPenalty
            penalties = obj.compute_penalties();
            grad{1} = -dLdz*x'/N - penalties{1}; % -dL/dW
            grad{2} = -mean(dLdz, 2) - penalties{2}; % -dL/db
         else
            grad{1} = -dLdz*x'/N;
            grad{2} = -mean(dLdz, 2);
         end
      end
   end
   
   methods (Abstract)
      [dLdz, y] = dLdz(obj, x, t)
   end
   
end

