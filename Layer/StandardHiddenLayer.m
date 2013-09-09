classdef StandardHiddenLayer < HiddenLayer & StandardLayer
   
   methods
      function obj = StandardHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardLayer(inputSize, outputSize, varargin{:});
      end
      
      function [grad, dLdx] = backprop(obj, x, y, dLdy)
         N = size(x, 2);
         dLdz = dLdy.*obj.dydz(y);
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
      value = dydz(~, y)
   end
   
end

