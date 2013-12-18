classdef Conv2DLayer < ParamsFunctions & ConvLayer
   % A convolution layer for multiple channels of 2D signals.
   
   properties
      % params = {W, b} with W ~ nF x 1 x C x fR x fC and b ~ nF x 1
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
         if isempty(obj.initScale)
            obj.initScale = .05;
         end
         obj.params{1} = obj.initScale*obj.gpuState.randn(obj.nFilters, 1, obj.nChannels, ...
                                                            obj.filterRows, obj.filterCols);
         if strcmp(obj.initType, 'relu')
            obj.params{2} = obj.gpuState.ones(obj.nFilters, 1);
         else
            obj.params{2} = obj.gpuState.zeros(obj.nFilters, 1);
         end
      end
      
      function y = feed_forward(obj, x, ~)
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
                                    iC - obj.filterCols + 1);
         
         for i = 1:(iR - obj.filterRows + 1)
            for j = 1:(iC - obj.filterCols + 1)
               xSample = permute(x(:,:,i:i+obj.filterRows-1, j:j+obj.filterCols-1), ...
                                                      [5, 2, 1, 3, 4]); % 1 X N x C x fR x fC
               Wx(:,:,i,j) = sum(sum(sum(bsxfun(@times, xSample, obj.params{1}), 5), 4), 3);
            end
         end
      end
      
      function [grad, dLdx] = backprop(obj, x, dLdy)
         % dLdy ~ nF x N x (iR - fR + 1) x (iC - fC + 1)
         % dLdx ~ C x N x iR x iC
         yRows = obj.inputRows - obj.filterRows + 1;         
         yCols = obj.inputCols - obj.filterCols + 1;
         grad{2} = mean(sum(sum(dLdy, 4), 3), 2);
         grad{1} = obj.gpuState.zeros(size(obj.params{1})); % nF x 1 x C x fR x fC
         dLdy = permute(dLdy, [1, 2, 5, 3, 4]); % nF x N x 1 x yRows x yCols
         for i = 1:obj.filterRows
            for j = 1:obj.filterCols
               xSample = permute(x(:,:,i:i+yRows-1, j:j+yCols-1), [5, 2, 1, 3, 4]); % 1 x N x C x yRows x yCols
               grad{1}(:,:,:,i,j) = mean(sum(sum(...
                        bsxfun(@times, xSample, dLdy), 5), 4), 2); % nF x 1 x C
            end
         end
         
         [nF, N, ~, ~, ~] = size(dLdy);
         dLdx = obj.gpuState.zeros(1, N, obj.nChannels, obj.inputRows, obj.inputCols); % 1 x N x C x iR x iC 
         dLdy = cat(4, obj.gpuState.zeros(nF, N, 1, obj.filterRows-1, yCols), ...
                              dLdy, ...
                              obj.gpuState.zeros(nF, N, 1, obj.filterRows-1, yCols)); % nF x N x 1 x (yRows + fRows - 1) x yCol
         dLdy = cat(5, ...
             obj.gpuState.zeros(nF, N, 1, obj.inputRows+obj.filterRows-1, obj.filterCols-1), ...
             dLdy, ...
             obj.gpuState.zeros(nF, N, 1, obj.inputRows+obj.filterRows-1, obj.filterCols-1)); % nF x N x 1 x (yRows + fRows - 1) x (yCol + fCol - 1)
         WFlipped = flip(flip(obj.params{1}, 5), 4);
         for i = 1:obj.inputRows
            for j = 1:obj.inputCols
               dLdySeg = dLdy(:,:,:,i:i+obj.filterRows-1, j:j+obj.filterCols-1); % nF x N x 1 x fR x fC
               dLdx(:,:,:,i,j) = sum(sum(sum(bsxfun(@times, dLdySeg, WFlipped), 5), 4), 1); % 1 x N x C
            end
         end
         dLdx = permute(dLdx, [3, 2, 4, 5, 1]);
      end      
   end
end

