classdef Conv1DHiddenLayer < HiddenLayer & ParamsLayer & ReuseValsLayer
   % A convolution hidden layer for multiple channels of 1D signals with
   % max pooling. No nonlinearity is applied after pooling. The user should
   % follow with a NoParams nonlinearity layer.
   
   properties
      params % {W, b} with W ~ nF x 1 x C x fS and b ~ nF x 1
      prePoolVal
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
         obj = obj@ParamsLayer(varargin{:});
         obj = obj@ReuseValsLayer(varargin{:});
         obj.inputSize = inputSize;
         obj.nChannels = nChannels;
         obj.filterSize = filterSize;
         obj.nFilters = nFilters;
         obj.poolSize = poolSize;
         obj.outputSize = ceil((inputSize - filterSize + 1)/poolSize);
      end
      
      function init_params(obj)
         obj.params{1} = 2*obj.initScale*obj.gpuState.rand(obj.nFilters, 1, obj.nChannels, ...
                                                            obj.filterSize) - obj.initScale;
         if strcmp(obj.initType, 'relu')
            obj.params{2} = 10*obj.initScale*obj.gpuState.rand(obj.nFilters, 1);
         else
            obj.params{2} = obj.gpuState.zeros(obj.nFilters, 1);
         end
      end
      
      function y = feed_forward(obj, x)
         % x ~ C x N x X
         % y ~ nF x N x oS
         z = obj.compute_z(x); % nF x N x (X - fS + 1)
         [y, prePool] = obj.max_pooling(z); 
         
         if obj.isReuseVals
         	obj.prePoolVal = prePool;
         end
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

      function [y, prePool] = max_pooling(obj, z)
         [nF, N, yHatSize] = size(z);
         paddingSize = mod(yHatSize, obj.poolSize);
         if paddingSize == 0
            prePool = reshape(z, nF, N, obj.poolSize, []); % nF x N x poolSize x oS
         else % pad with NaN values
            prePool = obj.gpuState.nan(nF, N, obj.poolSize, (yHatSize + paddingSize)/obj.poolSize);
         end
         y = max(prePool, [], 3);
         y = permute(y, [1, 2, 4, 3]); % nF x N x oS  (permute puts singleton dimension in back)
      end
      
      function [grad, dLdx, y] = backprop(obj, x, y, dLdy)
         % dLdy ~ nF x N x oS
         % z ~ nF x N x (X - fS + 1)
         if obj.isReuseVals
            prePool = obj.prePoolVals;
         else
            z = obj.compute_z(x);
            [~, prePool] = obj.max_pooling(z);
         end
         
         [nF, N] = size(y);
         zSize = obj.inputSize - obj.filterSize + 1;
         mask = obj.gpuState.make_numeric(bsxfun(@eq, permute(y, [1 2 4 3]), prePool) ...
                                                         & ~isnan(prePool)); % nF x N x poolSize x oS
         dLdz = bsxfun(@times, permute(dLdy, [1, 2, 4, 3]), mask);
         dLdz = reshape(dLdz, nF, N, []);
         dLdz = dLdz(:,:,1:zSize); % nF x N x (X - fS + 1)
         
         grad{2} = mean(sum(dLdz, 3), 2);
         
         grad{1} = obj.gpuState.zeros(size(obj.params{1})); % nF x 1 x C x fS
         for i = 1:obj.filterSize
            xSeg = permute(x(:,:,i:zSize + i - 1), [4, 2, 1, 3]); % 1 x N x C x zSize
            grad{1}(:,:,:,i) = mean(sum(bsxfun(@times, xSeg, permute(dLdz, [1, 2, 4, 3])), 4), 2); % nF x 1 x C
         end
         
         dLdx = obj.gpuState.zeros(size(x)); % C x N x X
         for i = 1:zSize
            dLdzVal = permute(dLdz(:,:,i), [4, 2, 3, 1]); % 1 x N x 1 x nF
            dLdx(:,:,i:i+obj.filterSize-1) = dLdx(:,:,i:i+obj.filterSize-1) + ...
                         sum(bsxfun(@times, dLdzVal, permute(obj.params{1}, [3, 2, 4, 1])), 4); % C x N x fS 
         end
      end      

%       function Dy = compute_Dy(obj, x, y)
%          % pass
%       end
%       
%       function D2y = compute_D2y(obj, x, y, Dy)
%          % pass
%       end
   end
end

