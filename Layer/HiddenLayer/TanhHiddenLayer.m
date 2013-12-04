classdef TanhHiddenLayer < StandardHiddenLayer
   
   properties
      isLocallyLinear = false
   end
   
   methods
      function obj = TanhHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardHiddenLayer(inputSize, outputSize, varargin{:});
      end
      
      function [y, ffExtras] = feed_forward(obj, x)
         % Using robust implementation from "Neural Networks Tricks of the
         % Trade" 2nd Edition, Ch 11
         z = obj.compute_z(x);
         v = exp(-2*z);
         u = 2./(1 + v);
         y = u - 1;
         ffExtras = {v, u};
      end
      
      function value = compute_Dy(~, ffExtras, ~)
         [v, u] = ffExtras{:};
         value = v.*u.*u;
      end
      
      function value = compute_D2y(~, ~, y, Dy)
         value = -2*y.*Dy;
      end
   end
end

