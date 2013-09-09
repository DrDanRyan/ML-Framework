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
      
      function [grad, dLdx] = backprop(obj, x, y, dLdy)
         grad{1} = obj.gpuState.zeros(obj.outputSize, obj.inputSize, obj.k); % -dL/dW
         grad{2} = obj.gpuState.zeros(obj.outputSize, 1, obj.k); % -dL/db
         N = size(x, 2);
         dLdx = obj.gpuState.zeros(size(x));
         
         z = obj.compute_z(x);
         mask = obj.gpuState.make_numeric((bsxfun(@eq, z, y)));
         
         for idx = 1:obj.k
            dLdz = dLdy.*mask(:,:,idx);
            grad{1}(:,:,idx) = -dLdz*x'/N; 
            grad{2}(:,:,idx) = -mean(dLdz, 2); 
            dLdx = dLdx + obj.params{1}(:,:,idx)'*dLdz;
         end
      end
      
      function z = compute_z(obj, x)
         z = obj.gpuState.zeros(obj.outputSize, size(x,2), obj.k);
         W = obj.params{1};
         b = obj.params{2};
         for idx = 1:obj.k
            z(:,:,idx) = bsxfun(@plus, W(:,:,idx)*x, b(:, :, idx));
         end
      end
   end
   
end

