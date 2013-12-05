classdef ReluHiddenLayer < StandardHiddenLayer
   
   properties
      isLocallyLinear = true
   end
   
   methods
      function obj = ReluHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardHiddenLayer(inputSize, outputSize, varargin{:});
      end
      
      function init_params(obj)
         obj.params{1} = matrix_init(obj.outputSize, obj.inputSize, obj.initType, ...
                                          obj.initScale, obj.gpuState);
         obj.params{2} = 10*obj.initScale*obj.gpuState.ones(obj.outputSize, 1);
      end
      
      function y = feed_forward(obj, x)
         z = obj.compute_z(x);
         y = max(0, z);
      end
      
      function value = compute_Dy(~, ~, y)
         value = obj.gpuState.make_numeric(y > 0);
      end         
   end   
end

