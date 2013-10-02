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
      
      function [grad, dLdx, gradVariance, nonZeroTerms] = ...
            backprop_with_variance(obj, x, y, dLdy)
         % Penalty terms are not used
         N = size(x, 2);
         dLdz = dLdy.*obj.dydz(y);
         dLdx = obj.params{1}'*dLdz;
         
         grad{1} = -dLdz*x'/N;
         grad{2} = -mean(dLdz, 2);
         
         gradVariance{1} = (dLdz.^2)*(x.^2)'/N;
         gradVariance{2} = mean(dLdz.^2, 2);
         
         nonZero_dLdz = obj.gpuState.make_numeric(dLdz~=0);
         nonZero_xTrans = obj.gpuState.make_numeric(x'~=0);
         nonZeroTerms{1} = nonZero_dLdz*nonZero_xTrans;
         nonZeroTerms{2} = sum(nonZero_dLdz, 2);
      end
   end
   
   methods (Abstract)
      value = dydz(~, y)
   end
   
end

