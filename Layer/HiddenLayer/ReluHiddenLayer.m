classdef ReluHiddenLayer < StandardHiddenLayer
   
   properties
      nonlinearity = @relu; % relu(x) = max(0, x) REctified Linear Unit
   end
   
   methods
      function obj = ReluHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardHiddenLayer(inputSize, outputSize, varargin{:});
      end
      
      function value = compute_Dy(~, ~, y)
         value = single(y >= 0);
      end
      
      function value = compute_D2y(obj, ~, y)
         value = obj.gpuState.zeros(size(y));
      end
         
   end
   
end

