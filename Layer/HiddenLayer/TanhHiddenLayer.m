classdef TanhHiddenLayer < StandardLayer & HiddenLayer
   
   properties
      Dy
      % isLocallyLinear = false
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
            obj.Dy = v.*u.*u;
            obj.Dy(isnan) = 0;
         end
      end
      
      function [grad, dLdx] = backprop(obj, x, y, dLdy)
         dLdz = obj.Dy.*dLdy;
         obj.Dy = [];
         [grad, dLdx] = obj.grad_from_dLdz(x, dLdz);
      end
      
%       function value = compute_D2y(~, ~, y, Dy)
%          value = -2*y.*Dy;
%       end
   end
end

