classdef TanhHiddenLayer < StandardHiddenLayer
   
   properties
      nonlinearity = @tanh; 
   end
   
   methods
      function obj = TanhHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardHiddenLayer(inputSize, outputSize, varargin{:});
      end
      
      function value = compute_dydz(~, ~, y)
         value = (1 + y).*(1 - y);
      end
   end
   
end

