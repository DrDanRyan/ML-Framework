classdef LinearHiddenLayer < StandardHiddenLayer
   % A simple linear layer.
   
   properties
      isLocallyLinear = true
   end
   
   methods
      function obj = LinearHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardHiddenLayer(inputSize, outputSize, varargin{:});
      end
      
      function [y, z] = feed_forward(obj, x)
         z = obj.compute_z(x);
         y = z;
      end
      
      function value = compute_Dy(obj, ~, y)
         value = obj.gpuState.ones(size(y));
      end     
      
      function value = compute_D2y(obj, ~, y, ~)
         value = obj.gpuState.zeros(size(y));
      end
   end
end

