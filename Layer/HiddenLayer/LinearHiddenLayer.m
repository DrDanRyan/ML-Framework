classdef LinearHiddenLayer < StandardHiddenLayer
   % A simple linear layer.
   
   properties
      isLocallyLinear = true
   end
   
   methods
      function obj = LinearHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardHiddenLayer(inputSize, outputSize, varargin{:});
      end
      
      function [y, ffExtras] = feed_forward(obj, x)
         y = obj.compute_z(x);
         ffExtras = [];
      end
      
      function value = compute_Dy(obj, ~, y)
         value = obj.gpuState.ones(size(y));
      end     
      
      function value = compute_D2y(obj, ~, y, ~)
         value = obj.gpuState.zeros(size(y));
      end
   end
end

