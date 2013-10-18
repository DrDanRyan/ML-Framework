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
%          obj.params{1} = orthonorm(obj.outputSize, obj.inputSize, obj.k, ...
%                                     obj.initScale, obj.gpuState);
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
      
      function [grad, dLdx, dydz] = backprop(obj, x, y, dLdy)
         % L1 and L2 penalties are not implemented for MaxoutHiddenLayer
         [L1, N] = size(x);         
         z = obj.compute_z(x);
         dydz = obj.gpuState.make_numeric((bsxfun(@eq, z, y))); % L2 x N x k
         dLdz = bsxfun(@times, dLdy, dydz); % dimensions are L2 x N x k
         dLdx = sum(pagefun(@mtimes, permute(obj.params{1}, [2, 1, 3]), dLdz), 3);
         
         switch obj.gradType
            case 'averaged'
               grad{1} = pagefun(@mtimes, dLdz, x')/N; % L2 x L1 x k
               grad{2} = mean(dLdz, 2); % L2 x 1 x k
            case 'sparse'               
               nonZero_dLdz = obj.gpuState.make_numeric(dLdz ~= 0);
               nonZero_xTrans = obj.gpuState.make_numeric(x' ~= 0);
               total_nonZero_dLdw = pagefun(@mtimes, nonZero_dLdz, nonZero_xTrans);
               total_nonZero_dLdz = sum(nonZero_dLdz, 2);

               total_nonZero_dLdw(total_nonZero_dLdw == 0) = 1; % Prevents dividing by zero below
               total_nonZero_dLdz(total_nonZero_dLdz == 0) = 1; % Prevents dividing by zero below
               grad{1} = pagefun(@mtimes, dLdz, x')./total_nonZero_dLdw;
               grad{2} = sum(dLdz, 2)./total_nonZero_dLdz;
            case 'raw'
               L2 = obj.outputSize;
               x4D = reshape(x, [1, L1, 1, N]);
               grad{2} = reshape(dLdz, [L2, 1, obj.k, N]);
               grad{1} = bsxfun(@times, grad{2}, x4D);            
         end
      end
      
      function z = compute_z(obj, x)
         % z has dimensions L2 x N x k
         z = pagefun(@mtimes, obj.params{1}, x);
         z = bsxfun(@plus, z, obj.params{2});
      end
      
   end
   
end

