classdef LinearHiddenLayer < StandardHiddenLayer
   % A simple linear layer.
   
   properties
      isLocallyLinear = true
   end
   
   methods
      function obj = LinearHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardHiddenLayer(inputSize, outputSize, varargin{:});
      end
      
      function y = feed_forward(obj, x)
         y = obj.compute_z(x);
      end
      
      function value = compute_Dy(obj, ~, y)
         value = obj.gpuState.ones(size(y));
      end     
   end
end

