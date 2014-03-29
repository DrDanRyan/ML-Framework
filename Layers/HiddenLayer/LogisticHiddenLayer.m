classdef LogisticHiddenLayer < StandardLayer & HiddenLayer
   % A standard hidden layer with logistic nonlinearity.
   
   properties
      dydz
   end
   
   methods
      function obj = LogisticHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardLayer(inputSize, outputSize, varargin{:});
      end
      
      function y = feed_forward(obj, x, isSave)
         z = obj.compute_z(x);
         y = 1./(1 + exp(-z));
         
         if nargin == 3 && isSave
            % more robust than using y*(1-y) on backprop pass
            obj.dydz = exp(-z).*y.*y;
         end             
      end
      
      function [grad, dLdx] = backprop(obj, x, ~, dLdy)
         dLdz = obj.dydz.*dLdy;
         obj.dydz = [];
         [grad, dLdx] = obj.grad_from_dLdz(x, dLdz);
      end
   end
end

