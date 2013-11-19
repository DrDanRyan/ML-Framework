classdef ReluHiddenLayer < StandardHiddenLayer
   
   properties
      isLocallyLinear = true
   end
   
   methods
      function obj = ReluHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardHiddenLayer(inputSize, outputSize, varargin{:});
      end
      
      function [y, ffExtras] = feed_forward(obj, x)
         z = obj.compute_z(x);
         y = max(0, z);
         ffExtras = [];
      end
      
      function value = compute_Dy(~, ~, y)
         value = obj.gpuState.make_numeric(y > 0);
      end
      
      function value = compute_D2y(obj, ~, y, ~)
         value = obj.gpuState.zeros(size(y));
      end
         
   end
   
end

