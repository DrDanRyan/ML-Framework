classdef LogisticHiddenLayer < StandardHiddenLayer
   
   properties
      nonlinearity = @sigm;
   end
   
   methods
      function obj = LogisticHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardHiddenLayer(inputSize, outputSize, varargin{:});
      end
      
      function value = compute_dydz(~, ~, y)
         value = y.*(1-y);
      end
   end
   
end

