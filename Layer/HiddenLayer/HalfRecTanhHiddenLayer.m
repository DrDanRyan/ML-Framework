classdef HalfRecTanhHiddenLayer < StandardLayer & HiddenLayer
   
   properties
      Dy
      % isLocallyLinear = false
   end
   
   methods
      function obj = HalfRecTanhHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardLayer(inputSize, outputSize, varargin{:});
      end
      
      function y = feed_forward(obj, x, isSave)
         % Using robust implementation from "Neural Networks Tricks of the
         % Trade" 2nd Edition, Ch 11
         z = obj.compute_z(x);
         v = exp(-2*z);
         u = 2./(1 + v);
         y = max(0, u - 1);         
         
         if nargin == 3 && isSave
            obj.Dy = v.*u.*u;
            obj.Dy(y == 0) = 0;
            obj.Dy(isnan(obj.Dy)) = 0;
         end
      end
      
      function [grad, dLdx] = backprop(obj, x, ~, dLdy)
         dLdz = obj.Dy.*dLdy;
         obj.Dy = [];
         [grad, dLdx] = obj.grad_from_dLdz(x, dLdz);
      end
   end
end

