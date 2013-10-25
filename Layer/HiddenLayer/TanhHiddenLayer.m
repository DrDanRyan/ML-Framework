classdef TanhHiddenLayer < StandardHiddenLayer
   
   properties
      nonlinearity = @tanh; 
   end
   
   methods
      function obj = TanhHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardHiddenLayer(inputSize, outputSize, varargin{:});
      end
      
      function value = compute_Dy(~, ~, y)
         value = (1 + y).*(1 - y);
      end
      
      function value = compute_D2y(~, ~, y)
         value = -2*y.*(1+y).*(1-y);
      end
   end
   
end

