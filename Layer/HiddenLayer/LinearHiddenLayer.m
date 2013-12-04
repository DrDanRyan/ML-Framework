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
         y = obj.compute_z(x);
         z = [];
      end
      
      function value = compute_Dy(obj, z, y)
         value = obj.gpuState.ones(size(y));
      end     
   end
end

