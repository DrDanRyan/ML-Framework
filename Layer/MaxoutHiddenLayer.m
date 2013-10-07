classdef MaxoutHiddenLayer < HiddenLayer & StandardLayer
   
   properties
      % params = {W, b} where W and b are 3-dimensional arrays
      nonlinearity % dummy property that doesnt get used... needed to inherit from StandardLayer
      k % number of linear units per maxout units (size of 3rd dimension of W and b)
   end
   
   methods
      function obj = MaxoutHiddenLayer(inputSize, outputSize, k, varargin)
         obj = obj@StandardLayer(inputSize, outputSize, varargin{:});
         obj.k = k;   
         obj.init_params();
      end
      
      function init_params(obj)
         obj.params{1} = obj.gpuState.zeros(obj.outputSize, obj.inputSize, obj.k);
         obj.params{2} = obj.gpuState.zeros(obj.outputSize, 1, obj.k);
         for idx = 1:obj.k
            obj.params{1}(:,:,idx) = matrix_init(obj.outputSize, obj.inputSize, obj.initType, ...
                                                      obj.initScale, obj.gpuState);
         end
      end
      
      function y = feed_forward(obj, x)
         z = obj.compute_z(x);
         y = max(z, [], 3);
      end
      
      function [grad, dLdx] = backprop(obj, x, y, dLdy, isAveraged)
         if nargin < 5
            isAveraged = true;
         end
         
         [L1, N] = size(x);         
         z = obj.compute_z(x);
         mask = obj.gpuState.make_numeric((bsxfun(@eq, z, y)));
         dLdz = bsxfun(@times, dLdy, mask);
         dLdx = sum(pagefun(@mtimes, permute(obj.params{1}, [2, 1, 3]), dLdz), 3);
         
         if isAveraged         
            grad{1} = pagefun(@mtimes, dLdz, x')/N;
            grad{2} = mean(dLdz, 2);
         else
            L2 = obj.outputSize;
            x4D = reshape(x, [1, L1, 1, N]);
            grad{2} = reshape(dLdz, [L2, 1, obj.k, N]);
            grad{1} = bsxfun(@times, grad{2}, x4D);            
         end
      end
      
      function z = compute_z(obj, x)
         z = pagefun(@mtimes, obj.params{1}, x);
         z = bsxfun(@plus, z, obj.params{2});
      end
   end
   
end

