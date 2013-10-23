classdef ReluHiddenLayer < StandardHiddenLayer
   
   properties
      nonlinearity = @relu; % relu(x) = max(0, x) REctified Linear Unit
   end
   
   methods
      function obj = ReluHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardHiddenLayer(inputSize, outputSize, varargin{:});
      end
      
      function value = dydz(~, ~, y)
         value = single(y >= 0);
      end
   end
   
end

