classdef InterferenceLayer < HiddenLayer & StandardLayer
   
   properties
      % params = {W, b, A, c}
      isLocallyLinear = false
      isDiagonalDy = false
   end
   
   methods
      function obj = InterferenceLayer(inputSize, outputSize, varargin)
         obj = obj@StandardLayer(inputSize, outputSize, varargin{:});
      end
      
      function init_params(obj)
         init_params@StandardLayer(obj);
         obj.params{3} = matrix_init(obj.outputSize, obj.outputSize, 'small positive', ...
                                          obj.initScale, obj.gpuState);
         obj.params{4} = matrix_init(obj.outputSize, 1, 'dense', ...
                                          obj.initScale, obj.gpuState);
      end
      
      function y = feed_forward(obj, x)
         z = obj.compute_z(x);
         y = obj.compute_y(z);
      end
      
      function [y, yHat, b, interference] = compute_y(obj, z)
         zMax = max(z, 1); % 1 x N
         z = bsxfun(@minus, z, zMax); % rescale for stability (L2 x N)
         b = bsxfun(@minus, obj.params{4}, zMax); % rescaled interference bias (L2 x N)
         yHat = exp(z);
         interference = obj.params{3}*yHat + exp(b);
         y = yHat./interference;
      end
      
      function [grad, dLdx, y] = backprop(obj, x, y, dLdy)
         N = size(x, 2);
         [Dy, yHat, b, interference] = obj.compute_Dy(x, y);
         dLdz = squeeze(pagefun(@mtimes, permute(Dy, [1 3 2]), permute(dLdy, [1 3 2]))); % L2 x N
         grad{1} = dLdz*x'/N;
         grad{2} = mean(dLdz, 2);
         dLdx = obj.params{1}'*dLdz;
         
         t1 = permute(-y.*dLdy./interference, [1 3 2]); % i x 1 x n
         t2 = shiftdim(yHat, -1); % 1 x j x n
         grad{3} = mean(bsxfun(@times, t1, t2), 3);
         grad{4} = mean(squeeze(t1).*exp(b), 2);
      end
      
      function [Dy, yHat, b, interference] = compute_Dy(obj, x, y)
         % returns D^n_{ij} as shape like j x n x i (L2 x N x L2)
         L2 = size(y, 1);
         z = obj.compute_z(x);
         [~, yHat, b, interference] = obj.compute_y(z);
         Dy = bsxfun(@times, permute(yHat, [1 2 3]), ...
                           permute(y./interference, [3 2 1]));
         Dy = bsxfun(@times, Dy, permute(obj.params{3}, [2 3 1]));
         id13 = permute(eye(L2), [1 3 2]);
         Dy = bsxfun(@times, y, id13) - Dy;
      end
      
      function value = compute_D2y(obj, x, y, Dy)
         % pass (need to implement in order to use CAE and MTC)
      end
      
      function increment_params(obj, delta)
         % require A=params{3} to have 1 on diagonals and nonnegative
         % everywhere else
         
         increment_params@StandardLayer(obj, delta);
         obj.params{3} = obj.params{3} + delta{3};
         obj.params{3} = max(0, obj.params{3}); % keep A nonnegative
         obj.params{3}(logical(obj.gpuState.eye(obj.outputSize))) = 1; % make sure diagonal(A) = 1
         
         obj.params{4} = obj.params{4} + delta{4};
      end     
      
   end
   
end

