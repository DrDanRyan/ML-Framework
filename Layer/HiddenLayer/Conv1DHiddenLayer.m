classdef Conv1DHiddenLayer < HiddenLayer
   % A convolution hidden layer for multiple channels of 1D signals with
   % max pooling. A tanh nonlinearity is used.
   
   properties
      params % {W, b} with W ~ nF x 1 x C x fS and b ~ nF x 1
      poolSize % (pS) number of units to maxpool over
      nFilters % (nF) number of convolution filters
      inputSize % (X) length of each 1D inputs signal
      nChannels % (C) number of input channels
      filterSize % (fS) length of the filter on each channel
      
      initScale % used for filter initialization
      gpuState
      outputSize % (oS) not specified by user, derived from other params at contruction
      isLocallyLinear = false
   end
   
   methods
      function obj = Conv1DHiddenLayer(inputSize, nChannels, filterSize, nFilters, poolSize, varargin)
         obj.inputSize = inputSize;
         obj.nChannels = nChannels;
         obj.filterSize = filterSize;
         obj.nFilters = nFilters;
         obj.poolSize = poolSize;
         obj.outputSize = ceil((inputSize - filterSize + 1)/poolSize);
         
         p = inputParser();
         p.addParamValue('initScale', .005);
         p.addParamValue('gpu', []);
         parse(p, varargin{:});
         obj.initScale = p.Results.initScale;
         obj.gpuState = GPUState(p.Results.gpu);
         obj.init_params();
      end
      
      function init_params(obj)
         obj.params{1} = 2*obj.initScale*obj.gpuState.rand(obj.nFilters, 1, obj.nChannels, ...
                                                            obj.filterSize) - obj.initScale;
         obj.params{2} = obj.gpuState.zeros(obj.nFilters, 1);
      end
      
      function [y, ffExtras] = feed_forward(obj, x)
         % x ~ C x N x X
         % y ~ nF x N x oS
         z = obj.compute_z(x); % nF x N x (X - fS + 1)
         v = exp(-2*z);
         u = 2./(1 + v);
         yHat = u - 1; % (robust tanh) nF x N x (X - fS + 1)
         [y, prePool] = obj.max_pooling(yHat); 
         ffExtras = {z, v, u, prePool};
%        if check_nan(z, u, y, prePool)
%            keyboard();
%        end
      end
      
      function z = compute_z(obj, x)
         Wx = obj.filter_activations(x); % nF x N x (X - fS + 1)
         z = bsxfun(@plus, Wx, obj.params{2});
      end
      
      function Wx = filter_activations(obj, x)
         % x ~ C x N x X
         % W ~ nF x 1 x C x fS
         % Wx ~ nF x N x (X - fS + 1)
         [~, N, X] = size(x);
         Wx = obj.gpuState.zeros(obj.nFilters, N, X - obj.filterSize + 1);
         
         for i = 1:(X - obj.filterSize + 1)
            xSeg = permute(x(:,:,i:i+obj.filterSize-1), [4, 2, 1, 3]); % 1 X N x C x fS
            Wx(:,:,i) = sum(sum(bsxfun(@times, xSeg, obj.params{1}), 4), 3);
         end
      end

      function [y, prePool] = max_pooling(obj, yHat)
         [nF, N, yHatSize] = size(yHat);
         paddingSize = mod(yHatSize, obj.poolSize);
         if paddingSize == 0
            prePool = reshape(yHat, nF, N, obj.poolSize, []); % nF x N x poolSize x oS
         else % pad with NaN values
            prePool = obj.gpuState.nan(nF, N, obj.poolSize, (yHatSize + paddingSize)/obj.poolSize);
         end
         y = max(prePool, [], 3);
         y = permute(y, [1, 2, 4, 3]); % nF x N x oS  (permute puts singleton dimension in back)
         
         if check_nan(prePool, y)
            keyboard();
         end
      end
      
      function [grad, dLdx, y] = backprop(obj, x, y, ffExtras, dLdy)
         % dLdy ~ nF x N x oS
         % z ~ nF x N x (X - fS + 1)
         [z, v, u, prePool] = ffExtras{:};
         [nF, N, zSize] = size(u);
         dyHatdz = u.*u.*v; % robust tanh derivative
         dyHatdz(isnan(dyHatdz)) = 0; % correct for extreme z values yielding NaN derivative
         
         mask = obj.gpuState.make_numeric(bsxfun(@eq, permute(y, [1 2 4 3]), prePool) ...
                                                         & ~isnan(prePool)); % nF x N x poolSize x oS
         dLdyHat = bsxfun(@times, permute(dLdy, [1, 2, 4, 3]), mask);
         dLdyHat = reshape(dLdyHat, nF, N, []);
         dLdyHat = dLdyHat(:,:,1:zSize); % nF x N x (X - fS + 1)
         
         dLdz = bsxfun(@times, dyHatdz, dLdyHat); % nF x N x (X - fS + 1)
         grad{2} = mean(sum(dLdz, 3), 2);
         
         grad{1} = obj.gpuState.zeros(size(obj.params{1})); % nF x 1 x C x fS
         zSize = size(u, 3); % X - fS + 1
         for i = 1:obj.filterSize
            xSeg = permute(x(:,:,i:zSize + i - 1), [4, 2, 1, 3]); % 1 x N x C x zSize
            grad{1}(:,:,:,i) = mean(sum(bsxfun(@times, xSeg, permute(z, [1, 2, 4, 3])), 4), 2); % nF x 1 x C
         end
         
         dLdx = obj.gpuState.zeros(size(x)); % C x N x X
         for i = 1:zSize
            zVal = permute(z(:,:,i), [4, 2, 3, 1]); % 1 x N x 1 x nF
            dLdx(:,:,i:i+obj.filterSize-1) = dLdx(:,:,i:i+obj.filterSize-1) + ...
                         sum(bsxfun(@times, zVal, permute(obj.params{1}, [3, 2, 4, 1])), 4); % C x N x fS 
         end
         
         if check_nan(grad{1}, grad{2}, dLdx, y)
            keyboard();
         end
      end      
      
      function gather(obj)
         obj.params{1} = gather(obj.params{1});
         obj.params{2} = gather(obj.params{2});
         obj.gpuState.isGPU = false;
      end
      
      function push_to_GPU(obj)
         obj.params{1} = single(gpuArray(obj.params{1}));
         obj.params{2} = single(gpuArray(obj.params{2}));
         obj.gpuState.isGPU = true;
      end
      
      function increment_params(obj, delta_params)
         obj.params{1} = obj.params{1} + delta_params{1};
         obj.params{2} = obj.params{2} + delta_params{2};
      end

      function value = compute_Dy(obj, dydyHat, y)
         % pass
      end
      
      function value = compute_D2y(obj, dydyHat, y, Dy)
         % pass
      end
   end
end

