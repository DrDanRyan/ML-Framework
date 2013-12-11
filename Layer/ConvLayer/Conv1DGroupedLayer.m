classdef Conv1DGroupedLayer < ParamsFunctions & ConvLayer
   % A convolution layer for multiple channels of 1D signals.
   
   properties
      % params = {W, b} with W ~ nG x 1 x FPG x RF x fS and b ~ nG x 1 x FPG
      nGroups % (G) number of distinct receptive field groups
      RFSize % (RF) size of the receptive field for each group
      FPG    % number of filters per group
      inputSize % (X) length of each 1D inputs signal
      nChannels % (C) number of input channels
      filterSize % (fS) length of the filter on each channel
      groupAssignments % a G x C logical matrix showing group membership
      
      % Note total number of filters: nF = nGroups*FPG and each filter acts
      % on RFSize channels.
   end
   
   methods
      function obj = Conv1DGroupedLayer(inputSize, nChannels, filterSize, nGroups, RFSize, FPG, varargin)
         obj = obj@ParamsFunctions(varargin{:});
         obj.inputSize = inputSize;
         obj.nChannels = nChannels;
         obj.filterSize = filterSize;
         obj.nGroups = nGroups;
         obj.RFSize = RFSize;
         obj.FPG = FPG;
         obj.init_params();
      end
      
      function init_params(obj)
         if isempty(obj.initScale)
            obj.initScale = .05;
         end
         obj.params{1} = obj.initScale*obj.gpuState.randn(obj.nGroups, 1, obj.FPG, ...
                                                            obj.RFSize, obj.filterSize);
         if strcmp(obj.initType, 'relu')
            obj.params{2} = obj.gpuState.ones(obj.nGroups, obj.FPG);
         else
            obj.params{2} = obj.gpuState.zeros(obj.nGroups, 1, obj.FPG);
         end
      end
      
      function y = feed_forward(obj, x)
         % x ~ C x N x X
         % Wx ~ nG x N x FPG x cS (where cS = (X - fS + 1))
         % y ~ nF x N x cS
         Wx = obj.filter_activations(x);
         y = bsxfun(@plus, Wx, obj.params{2});
         y = permute(y, [1, 3, 2, 4]);
         [nG, FPG, N, cS] = size(y); %#ok<PROP>
         y = reshape(y, [nG*FPG, N, cS]); %#ok<PROP>
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
            xSeg = permute(x(:,:,i:ySize + i - 1), [4, 2, 1, 3]); % 1 x N x C x Size
            grad{1}(:,:,:,i) = mean(sum(bsxfun(@times, xSeg, permute(dLdy, [1, 2, 4, 3])), 4), 2); % nF x 1 x C
         end
         
         [nF, N, ~] = size(dLdy);
         dLdx = obj.gpuState.zeros(obj.inputSize, N, obj.nChannels); % X x N x C (need to permute before returning)
         dLdyPadded = cat(4, obj.gpuState.zeros(nF, N, 1, obj.filterSize-1), ...
                              permute(dLdy, [1, 2, 4, 3]), ...
                              obj.gpuState.zeros(nF, N, 1, obj.filterSize-1)); % nF x N x 1 x (X + fS - 1)
         for i = 1:obj.inputSize
            dLdySeg = dLdyPadded(:,:,:,i:i+obj.filterSize-1); % nF x N x 1 x fS
            dLdx(i,:,:) = sum(sum(bsxfun(@times, dLdySeg, flip(obj.params{1}, 4)), 4), 1); % 1 x N x C
         end
         dLdx = permute(dLdx, [3, 2, 1]);
      end      
   end
end

