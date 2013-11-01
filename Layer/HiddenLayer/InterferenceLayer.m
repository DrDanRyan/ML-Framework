classdef InterferenceLayer < HiddenLayer & StandardLayer
   
   properties
      % params = {W, b, A, c}
   end
   
   methods
      function obj = InterferenceLayer(inputSize, outputSize, varargin)
         obj = obj@StandardLayer(inputSize, outputSize, varargin{:});
      end
      
      function init_params(obj)
         init_params@StandardLayer(obj);
         obj.params{3} = matrix_init(obj.outputSize, obj.inputSize, 'small positive', ...
                                          obj.initScale, obj.gpuState);
         obj.params{4} = matrix_init(obj.outputSize, 1, 'near one', ...
                                          obj.initScale, obj.gpuState);
      end
      
      
   end
   
end

