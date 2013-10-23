classdef LinearHiddenLayer < StandardHiddenLayer
   % A simple linear layer. Useful for constructing a MaxoutAutoEncoder.
   
   properties
      nonlinearity
   end
   
   methods
      function obj = LinearHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardHiddenLayer(inputSize, outputSize, varargin{:});
      end
      
      function y = feed_forward(obj, x) % overide to prevent unnecessary identity arrayfun
         y = obj.compute_z(x);
      end
      
      function value = compute_dydz(obj, ~, y)
         value = obj.gpuState.ones(size(y));
      end     
   end
end

