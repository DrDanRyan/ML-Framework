classdef StandardLayer < ParamsLayer & RegularizationFunctions & ReuseValsLayer
   % A mixin that provides basic functionality for a standard layer
   % consisting of a linear layer (z = W*x + b) followed by a 
   % nonlinear function (y = f(z)).
   
   properties
      % params = {W, b}
      inputSize
      outputSize
   end
   
   methods
      function obj = StandardLayer(inputSize, outputSize, varargin)       
         obj = obj@ParamsLayer(varargin{:});
         obj = obj@RegularizationFunctions(varargin{:});
         obj = obj@ReuseValsLayer(varargin{:});
         obj.inputSize = inputSize;
         obj.outputSize = outputSize;
         obj.init_params();
      end
      
      function init_params(obj)
         obj.params{1} = matrix_init(obj.outputSize, obj.inputSize, obj.initType, ...
                                          obj.initScale, obj.gpuState);
         obj.params{2} = obj.gpuState.zeros(obj.outputSize, 1);
      end
      
      function increment_params(obj, delta)
         increment_params@ParamsLayer(obj, delta);
         if ~isempty(obj.maxFanIn)
            obj.impose_fanin_constraint();
         end
      end 
      
      function value = compute_z(obj, x)
         value = bsxfun(@plus, obj.params{1}*x, obj.params{2});
      end
      
      function grad = grad_from_dLdz(obj, x, dLdz)
         grad{1} = dLdz*x'/N;
         grad{2} = mean(dLdz, 2);
         
         if obj.isPenalty
            penalties = obj.compute_penalties();
            grad{1} = grad{1} + penalties{1};
            grad{2} = grad{2} + penalties{2};
         end
      end
      
   end
end

