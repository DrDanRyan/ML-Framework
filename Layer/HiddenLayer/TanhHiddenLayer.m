classdef TanhHiddenLayer < StandardHiddenLayer
   
   properties
      isLocallyLinear = false
      isDiagonalDy = true
   end
   
   methods
      function obj = TanhHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardHiddenLayer(inputSize, outputSize, varargin{:});
      end
      
      function [y, z] = feed_forward(obj, x)
         % Using robust implementation from "Neural Networks Tricks of the
         % Trade" 2nd Edition, Ch 11
         z = obj.compute_z(x);
         v = exp(-2*z);
         u = 2./(1 + v);
         y = u - 1;
      end
      
      function value = compute_Dy(~, z, ~)
         v = exp(-2*z);
         u = 2./(1 + v);
         value = v.*u.*u;
      end
      
      function value = compute_D2y(~, ~, y, Dy)
         value = -2*y.*Dy;
      end
   end
   
end

