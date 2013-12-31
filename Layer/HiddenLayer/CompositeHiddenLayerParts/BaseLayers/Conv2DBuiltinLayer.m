classdef Conv2DBuiltinLayer < ParamsFunctions & matlab.mixin.Copyable
   % A convolution layer for multiple channels of 1D signals.
   
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
      function obj = Conv2DBuiltinLayer(inputRows, inputCols, nChannels, filterRows, ...
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
         if isempty(obj.initScale)
            obj.initScale = 1/(obj.filterRows*obj.filterCols*obj.nChannels);
         end
         obj.params{1} = 2*obj.initScale*obj.gpuState.rand(obj.nFilters, obj.nChannels, 1, ...
                              obj.filterRows, obj.filterCols) - obj.initScale;
         if strcmp(obj.initType, 'relu')
            obj.params{2} = obj.initScale*obj.gpuState.ones(obj.nFilters, 1);
         else
            obj.params{2} = obj.gpuState.zeros(obj.nFilters, 1);
         end
      end
      
      function y = feed_forward(obj, x, ~)
         % x ~ C x N x iR x iC
         % y ~ nF x N x (iR - fR + 1) x (iC - fC + 1)
         Wx = obj.filter_activations(x);
         y = bsxfun(@plus, Wx, obj.params{2});
      end
      
      function Wx = filter_activations(obj, x)
         % x ~ C x N x iR x iC
         % W ~ nF x C x 1 x fR x fC
         % Wx ~ nF x N x (iR - fR + 1) x (iC - fC + 1)
         
         [~, N, iR, iC] = size(x);
         Wx = obj.gpuState.zeros(obj.nFilters, N, iR - obj.filterRows + 1, ...
                                    iC - obj.filterCols + 1);                         
         for i = 1:obj.nFilters
            Wx(i,:,:,:) = shiftdim(convn(x, shiftdim(obj.params{1}(i,:,:,:,:), 1), 'valid'), 1);
         end
      end
      
      function [grad, dLdx] = backprop(obj, x, dLdy)
         % dLdy ~ nF x N x (iR - fR + 1) x (iC - fC + 1)
         % dLdx ~ C x N x iR x iC
         grad{2} = mean(sum(sum(dLdy, 4), 3), 2);
         grad{1} = obj.gpuState.zeros(size(obj.params{1})); % nF x C x 1 x fR x fC
         [nF, N, ~, ~] = size(dLdy);
         
         for i = 1:obj.nFilters
            grad{1}(i,:,:,:,:) = ...
               flip(flip(flip(convn(x, flip(flip(flip(dLdy(i,:,:,:), 2), 3), 4), 'valid'), 1), 3), 4)/N;
         end

         dLdyPadded = obj.gpuState.zeros(nF, 2*obj.nChannels-1, N, ...
                              obj.inputRows+obj.filterRows-1, obj.inputCols+obj.filterCols-1);
         dLdyPadded(:,obj.nChannels,:,obj.filterRows:obj.inputRows, obj.filterCols:obj.inputCols) ...
                  = permute(dLdy, [1, 5, 2, 3, 4]);
         dLdx = shiftdim(convn(dLdyPadded, flip(flip(flip(flip(obj.params{1}, 1), 2), 4), 5), 'valid'), 1);
      end      
   end
end

