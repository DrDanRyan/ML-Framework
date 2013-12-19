classdef Conv2DLayer < ParamsFunctions & ConvLayer
   % A convolution layer for multiple channels of 2D signals.
   
   properties
      % params = {W, b} with W ~ nF x C x 1 x fR x fC and b ~ nF x 1
      nFilters % (nF) number of convolution filters
      inputRows % (iR) width of the 2D input signal
      inputCols % (iC) columns of the 2D input signal
      nChannels % (C) number of input channels
      filterRows % (fR)
      filterCols % (fC)
   end
   
   methods
      function obj = Conv2DLayer(inputRows, inputCols, nChannels, filterRows, ...
                                    filterCols, nFilters, varargin)
         obj = obj@ParamsFunctions(varargin{:});
         obj.inputRows = inputRows;
         obj.inputCols = inputCols;
         obj.nChannels = nChannels;
         obj.filterRows = filterRows;
         obj.filterCols = filterCols;
         obj.nFilters = nFilters;
         obj.init_params();
      end
      
      function init_params(obj)
         radius = 1/(obj.filterRows*obj.filterCols*obj.nChannels);
         obj.params{1} = 2*radius*obj.gpuState.rand(obj.nFilters, obj.nChannels, 1, ...
                              obj.filterRows, obj.filterCols) - radius;
         if strcmp(obj.initType, 'relu')
            obj.params{2} = radius*obj.gpuState.ones(obj.nFilters, 1);
         else
            obj.params{2} = obj.gpuState.zeros(obj.nFilters, 1);
         end
      end
      
      function y = feed_forward(obj, x, ~)
         % x ~ C x N x iR x iC
         % y ~ nF x N x yR x yC
         Wx = obj.filter_activations(x);
         y = bsxfun(@plus, Wx, obj.params{2});
      end
      
      function Wx = filter_activations(obj, x)
         % x ~ C x N x iR x iC
         % W ~ nF x C x 1 x fR x fC
         % Wx ~ nF x N x yR x yC
         [~, N, iR, iC] = size(x);
         yRows = iR - obj.filterRows + 1;
         yCols = iC - obj.filterCols + 1;
         Wx = obj.gpuState.zeros(obj.nFilters, 1, N, yRows, yCols);
         
         for i = 1:yRows
            for j = 1:yCols
               xSample = shiftdim(x(:,:,i:i+obj.filterRows-1, j:j+obj.filterCols-1), -1); % 1 X C x N x fR x fC
               Wx(:,:,:,i,j) = sum(sum(sum(bsxfun(@times, xSample, obj.params{1}), 5), 4), 2);
            end
         end
         Wx = permute(Wx, [1, 3, 4, 5, 2]); % remove channel dimension by shifting it to end
      end
      
      function [grad, dLdx] = backprop(obj, x, dLdy)
         % dLdy ~ nF x N x yR x yC
         % dLdx ~ C x N x iR x iC
         [nF, N, yRows, yCols] = size(dLdy);
         grad{2} = mean(sum(sum(dLdy, 4), 3), 2);
         grad{1} = obj.gpuState.zeros(size(obj.params{1})); % nF x C x 1 x fR x fC
         dLdy = permute(dLdy, [1, 5, 2, 3, 4]); % nF x 1 x N x yR x yC
         for i = 1:obj.filterRows
            for j = 1:obj.filterCols
               xSample = shiftdim(x(:,:,i:i+yRows-1, j:j+yCols-1), -1); % 1 x C x N x yR x yC
               grad{1}(:,:,1,i,j) = mean(sum(sum(bsxfun(@times, xSample, dLdy), 5), 4), 3); % nF x C
            end
         end
         clear xSample
         
         dLdx = obj.gpuState.zeros(1, obj.nChannels, N, obj.inputRows, obj.inputCols); % 1 x C x N x iR x iC 
         dLdyPadded = obj.gpuState.zeros(nF, 1, N, obj.inputRows+obj.filterRows-1, ...
                                             obj.inputCols+obj.filterCols-1);
         dLdyPadded(:,:,:,obj.filterRows:obj.inputRows, obj.filterCols:obj.inputCols) = dLdy;
         clear dLdy
         
         WFlipped = flip(flip(obj.params{1}, 5), 4);
         for i = 1:obj.inputRows
            for j = 1:obj.inputCols
               dLdySeg = dLdyPadded(:,:,:,i:i+obj.filterRows-1, j:j+obj.filterCols-1); % nF x 1 x N x fR x fC
               dLdx(:,:,:,i,j) = sum(sum(sum(bsxfun(@times, dLdySeg, WFlipped), 5), 4), 1); % 1 x C x N
            end
         end
         dLdx = shiftdim(dLdx, 1);
      end      
   end
end

