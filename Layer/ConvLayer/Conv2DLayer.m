classdef Conv2DLayer < ParamsFunctions & ConvLayer
   % A convolution layer for multiple channels of 1D signals.
   
   properties
      % params = {W, b} with W ~ nF x 1 x C x fR x fC and b ~ nF x 1
      nFilters % (nF) number of convolution filters
      inputRows % (iR) width of the 2D input signal
      inputColumns % (iC) columns of the 2D input signal
      nChannels % (C) number of input channels
      filterRows % (fR)
      filterComlumns % (fC)
   end
   
   methods
      function obj = Conv2DLayer(inputRows, inputColumns, nChannels, filterRows, ...
                                    filterColumns, nFilters, varargin)
         obj = obj@ParamsFunctions(varargin{:});
         obj.inputRows = inputRows;
         obj.inputColumns = inputColumns;
         obj.nChannels = nChannels;
         obj.filterRows = filterRows;
         obj.filterColumns = filterColumns;
         obj.nFilters = nFilters;
         obj.init_params();
      end
      
      function init_params(obj)
         if isempty(obj.initScale)
            obj.initScale = .05;
         end
         obj.params{1} = obj.initScale*obj.gpuState.randn(obj.nFilters, 1, obj.nChannels, ...
                                                            obj.filterRows, obj.filterColumns);
         if strcmp(obj.initType, 'relu')
            obj.params{2} = obj.gpuState.ones(obj.nFilters, 1);
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
         % x ~ C x N x iR x iC
         % W ~ nF x 1 x C x fR x fC
         % Wx ~ nF x N x 
         [~, N, iR, iC] = size(x);
         Wx = obj.gpuState.zeros(obj.nFilters, N, iR - obj.filterRows + 1, ...
                                    iC - obj.filterColumns + 1);
         
         for i = 1:(iR - obj.filterRows + 1)
            for j = 1:(iC - obj.filterColumns + 1)
               xSample = permute(x(:,:,i:i+obj.filterRows-1, j:j+obj.filterColumns-1), ...
                                                      [5, 2, 1, 3, 4]); % 1 X N x C x fR x fC
               Wx(:,:,i,j) = sum(sum(sum(bsxfun(@times, xSample, obj.params{1}), 5), 4), 3);
            end
         end
      end
      
      function [grad, dLdx] = backprop(obj, x, dLdy)
         % dLdy ~ nF x N x (iR - fR + 1) x (iC - fC + 1)
         yRows = obj.inputRows - obj.filterRows + 1;         
         yColumns = obj.inputColumns - obj.filterColumns + 1;
         grad{2} = mean(sum(sum(dLdy, 4), 3), 2);
         grad{1} = obj.gpuState.zeros(size(obj.params{1})); % nF x 1 x C x fR x fC
         for i = 1:obj.filterRows
            for j = 1:obj.filterColumns
               xSample = permute(x(:,:,i:i+yRows-1, j:j+yColumns-1), [5, 2, 1, 3, 4]); % 1 x N x C x yRows x yColumns
               grad{1}(:,:,:,i,j) = mean(sum(sum(...
                        bsxfun(@times, xSample, permute(dLdy, [1, 2, 5, 3, 4])), 5, 4)), 2); % nF x 1 x C
            end
         end
         
         [nF, N, ~] = size(dLdy);
         dLdx = obj.gpuState.zeros(1, N, obj.nChannels, obj.inputRows, obj.inputColumns); % 1 x N x C x iR x iC 
         dLdyPadded = cat(4, obj.gpuState.zeros(nF, N, 1, obj.filterSize-1), ...
                              permute(dLdy, [1, 2, 5, 3, 4]), ...
                              obj.gpuState.zeros(nF, N, 1, obj.filterSize-1)); % nF x N x 1 x (X + fS - 1)
         for i = 1:obj.inputSize
            dLdySeg = dLdyPadded(:,:,:,i:i+obj.filterSize-1); % nF x N x 1 x fS
            dLdx(i,:,:) = sum(sum(bsxfun(@times, dLdySeg, flip(obj.params{1}, 4)), 4), 1); % 1 x N x C
         end
         dLdx = permute(dLdx, [3, 2, 1]);
      end      
   end
end

