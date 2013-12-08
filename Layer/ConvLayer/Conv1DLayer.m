classdef Conv1DLayer < ParamsFunctions & ConvLayer
   % A convolution layer for multiple channels of 1D signals.
   
   properties
      % params = {W, b} with W ~ nF x 1 x C x fS and b ~ nF x 1
      nFilters % (nF) number of convolution filters
      inputSize % (X) length of each 1D inputs signal
      nChannels % (C) number of input channels
      filterSize % (fS) length of the filter on each channel
   end
   
   methods
      function obj = Conv1DLayer(inputSize, nChannels, filterSize, nFilters, varargin)
         obj = obj@ParamsFunctions(varargin{:});
         obj.inputSize = inputSize;
         obj.nChannels = nChannels;
         obj.filterSize = filterSize;
         obj.nFilters = nFilters;
         obj.init_params();
      end
      
      function init_params(obj)
         if isempty(obj.initScale)
            obj.initScale = .01;
         end
         
         obj.params{1} = 2*obj.initScale*obj.gpuState.rand(obj.nFilters, 1, obj.nChannels, ...
                                                            obj.filterSize) - obj.initScale;
         if strcmp(obj.initType, 'relu')
            obj.params{2} = obj.initScale*obj.gpuState.randn(obj.nFilters, 1) + ...
                                 4*obj.initScale*obj.gpuState.ones(obj.nFilters, 1);
         else
            obj.params{2} = obj.gpuState.zeros(obj.nFilters, 1);
         end
      end
      
      function y = feed_forward(obj, x)
         % x ~ C x N x X
         % y ~ nF x N x (X - fS + 1)
         Wx = obj.filter_activations(x);
         y = bsxfun(@plus, Wx, obj.params{2});
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
      
      function [grad, dLdx] = backprop(obj, x, dLdy)
         % dLdy ~ nF x N x (X - fS + 1)
         ySize = obj.inputSize - obj.filterSize + 1;         
         grad{2} = mean(sum(dLdy, 3), 2);
         grad{1} = obj.gpuState.zeros(size(obj.params{1})); % nF x 1 x C x fS
         for i = 1:obj.filterSize
            xSeg = permute(x(:,:,i:ySize + i - 1), [4, 2, 1, 3]); % 1 x N x C x zSize
            grad{1}(:,:,:,i) = mean(sum(bsxfun(@times, xSeg, permute(dLdy, [1, 2, 4, 3])), 4), 2); % nF x 1 x C
         end
         
         dLdx = obj.gpuState.zeros(size(x)); % C x N x X
         for i = 1:ySize
            dLdyVal = permute(dLdy(:,:,i), [4, 2, 3, 1]); % 1 x N x 1 x nF
            dLdx(:,:,i:i+obj.filterSize-1) = dLdx(:,:,i:i+obj.filterSize-1) + ...
                         sum(bsxfun(@times, dLdyVal, permute(obj.params{1}, [3, 2, 4, 1])), 4); % C x N x fS 
         end
      end      
   end
end

