classdef TanhHiddenLayer < StandardLayer & HiddenLayer
   % Standard hidden layer with hyperbolic tangent nonlinearity.
   
   properties
      dydz
   end
   
   methods
      function obj = TanhHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardLayer(inputSize, outputSize, varargin{:});
      end
      
      function y = feed_forward(obj, x, isSave)
         % Using robust implementation from "Neural Networks Tricks of the
         % Trade" 2nd Edition, Ch 11
         z = obj.compute_z(x);
         v = exp(-2*z);
         u = 2./(1 + v);
         y = u - 1;
         
         if nargin == 3 && isSave
            obj.dydz = v.*u.*u;
            obj.dydz(isnan(obj.dydz)) = 0;
         end
      end
      
      function [grad, dLdx] = backprop(obj, x, ~, dLdy)
         dLdz = obj.dydz.*dLdy;
         obj.dydz = [];
         [grad, dLdx] = obj.grad_from_dLdz(x, dLdz);
      end
   end
end

