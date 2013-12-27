classdef NDimLinearLayer < ParamsFunctions & matlab.mixin.Copyable & RegularizationFunctions
   
   properties
      % params = {W, b} where W and b are 3-dimensional arrays
      inputSize
      outputSize
      D % number of linear units per maxout unit (size of 3rd dimension of W and b)
   end
   
   methods
      function obj = NDimLinearLayer(inputSize, outputSize, D, varargin)
         obj = obj@ParamsFunctions(varargin{:});
         obj = obj@RegularizationFunctions(varargin{:});
         obj.inputSize = inputSize;
         obj.outputSize = outputSize;
         obj.D = D;   
         obj.init_params();
      end
      
      function init_params(obj)
         obj.params{2} = obj.gpuState.zeros(obj.outputSize, 1, obj.D);
         for idx = 1:obj.D
            obj.params{1}(:,:,idx) = matrix_init(obj.outputSize, obj.inputSize, obj.initType, ...
                                                      obj.initScale, obj.gpuState);
         end
      end
      
      function y = feed_forward(obj, x, ~)
         y = obj.compute_z(x);
      end
      
      function [grad, dLdx] = backprop(obj, x, dLdy)
         N = size(x, 2);         
         dLdx = sum(pagefun(@mtimes, permute(obj.params{1}, [2, 1, 3]), dLdy), 3);
         grad{1} = pagefun(@mtimes, dLdy, x')/N; % L2 x L1 x D
         grad{2} = mean(dLdy, 2); % L2 x 1 x D
         
         if obj.isPenalty
            penalties = obj.compute_penalties();
            grad{1} = grad{1} + penalties{1};
            grad{2} = grad{2} + penalties{2};
         end
      end
      
      function value = compute_z(obj, x)
         % z has dimensions L2 x N x D
         value = pagefun(@mtimes, obj.params{1}, x);
         value = bsxfun(@plus, value, obj.params{2});
      end
      
   end
end

