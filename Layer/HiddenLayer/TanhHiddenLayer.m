classdef TanhHiddenLayer < StandardHiddenLayer
   
   properties
      uVal
      vVal
      isLocallyLinear = false
   end
   
   methods
      function obj = TanhHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardHiddenLayer(inputSize, outputSize, varargin{:});
      end
      
      function y = feed_forward(obj, x)
         % Using robust implementation from "Neural Networks Tricks of the
         % Trade" 2nd Edition, Ch 11
         z = obj.compute_z(x);
         v = exp(-2*z);
         u = 2./(1 + v);
         y = u - 1;
         
         if obj.isReuseVals
            obj.uVal = u;
            obj.vVal = v;
         end
      end
      
      function Dy = compute_Dy(obj, x, ~)
         if obj.isReuseVals
            Dy = obj.vVal.*obj.uVal.*obj.uVal;
         else
            z = obj.compute_z(x);
            v = exp(-2*z);
            u = 2./(1 + v);
            Dy = v.*u.*u;
         end
         Dy(isnan(Dy)) = 0; % extreme values of z lead to NaN instead of 0 derivative
      end
      
      function value = compute_D2y(~, ~, y, Dy)
         value = -2*y.*Dy;
      end
   end
end

