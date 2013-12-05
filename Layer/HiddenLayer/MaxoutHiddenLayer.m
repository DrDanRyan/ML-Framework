classdef MaxoutHiddenLayer < HiddenLayer & ParamsLayer & RegularizationFunctions & ReuseValsLayer
   
   properties
      % params = {W, b} where W and b are 3-dimensional arrays
      inputSize
      outputSize
      D % number of linear units per maxout unit (size of 3rd dimension of W and b)
      zVal
      isLocallyLinear = true
   end
   
   methods
      function obj = MaxoutHiddenLayer(inputSize, outputSize, D, varargin)
         obj = obj@ParamsLayer(varargin{:});
         obj = obj@RegularizationFunctions(varargin{:});
         obj = obj@ReuseValsLayer(varargin{:});
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
      
      function [y, z] = feed_forward(obj, x)
         z = obj.compute_z(x);
         y = max(z, [], 3);
         
         if obj.isReuseVals
            obj.zVal = z;
         end
      end
      
      function [grad, dLdx, Dy] = backprop(obj, x, y, dLdy)
         N = size(x, 2);         
         Dy = obj.compute_Dy(x, y);
         dLdz = bsxfun(@times, dLdy, Dy); % dimensions are L2 x N x D
         dLdx = sum(pagefun(@mtimes, permute(obj.params{1}, [2, 1, 3]), dLdz), 3);
         grad{1} = pagefun(@mtimes, dLdz, x')/N; % L2 x L1 x D
         grad{2} = mean(dLdz, 2); % L2 x 1 x D
         
         if obj.isPenalty
            penalties = obj.compute_penalties();
            grad{1} = grad{1} + penalties{1};
            grad{2} = grad{2} + penalties{2};
         end
      end
      
      function value = compute_Dy(obj, x, y)
         if obj.isReuseVals
            z = obj.zVal;
         else
            z = obj.compute_z(x);
         end
         
         value = obj.gpuState.make_numeric((bsxfun(@eq, z, y))); % L2 x N x D
      end
      
      function value = compute_z(obj, x)
         % z has dimensions L2 x N x D
         value = pagefun(@mtimes, obj.params{1}, x);
         value = bsxfun(@plus, value, obj.params{2});
      end
   end
end

