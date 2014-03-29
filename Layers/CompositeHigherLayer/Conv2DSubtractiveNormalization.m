classdef Conv2DSubtractiveNormalization < matlab.mixin.Copyable
   % Performs subtractive and divisive contrast normalization over Gaussian
   % window and accross feature maps (see LeCun2010 "Convolutional networks
   % and applications in vision")
   
   properties
      windowSize % (wS) The Gaussian weighting window is a wS x wS square
      W % ~ 1 x 1 x wS x wS   weight coefficients for Gaussian weighting window
      gpuState
      
      % A 1 x 1 x rows x cols matrix the same size as a single input channel 
      % with edge correction coeffs that make gaussian filter keep sum of 1
      edgeFactor 
   end
   
   methods
      function obj = Conv2DSubtractiveNormalization(windowSize, channels, gpu)
         obj.windowSize = windowSize;
         if nargin < 3
            obj.gpuState = GPUState();
         else
            obj.gpuState = GPUState(gpu);
         end
         obj.W = obj.compute_weights(channels);
      end
      
      function y = feed_forward(obj, x, ~)
         radius = (obj.windowSize-1)/2;
         [C, N, rows, cols] = size(x);
         convTerm = obj.gpuState.nan(1, N, rows, cols);
         
         if isempty(obj.edgeFactor)
            isStoreEdgeFactor = true;
            obj.edgeFactor = obj.gpuState.ones(1, 1, rows, cols);
         else
            isStoreEdgeFactor = false;
         end

         for i = 1:rows
            for j = 1:cols
               if i < radius+1 || i > rows-radius || ... % Edge effects, need to truncate
                     j < radius+1 || j > cols-radius     % filter window
                  samp = x(:,:,max(1,i-radius):min(rows,i+radius), ...
                               max(1,j-radius):min(cols,j+radius));
                  topRoom = min(radius, i-1);
                  bottomRoom = min(radius, rows-i);
                  leftRoom = min(radius, j-1);
                  rightRoom = min(radius, cols-j);
                  center = radius+1;
                  truncW = obj.W(1,1,center-topRoom:center+bottomRoom, ...
                                     center-leftRoom:center+rightRoom);
                  if isStoreEdgeFactor
                     obj.edgeFactor(:,:,i,j) = C*sum(truncW(:));
                  end
                  convTerm(:,:,i,j) = sum(sum(sum(bsxfun(@times, ...
                     truncW/obj.edgeFactor(:,:,i,j), samp), 4), 3), 1);
               else   % Interior, can use full filter
                  samp = x(:,:,i-radius:i+radius, j-radius:j+radius);
                  convTerm(:,:,i,j) = sum(sum(sum(bsxfun(@times, obj.W, samp), 4), 3), 1);
               end
            end
         end
         y = bsxfun(@minus, x, convTerm);
      end
      
      function dLdx = backprop(obj, dLdy)
         radius = (obj.windowSize-1)/2;
         [~, N, rows, cols] = size(dLdy);
         convTerm = obj.gpuState.nan(1, N, rows, cols);
         
         for i = 1:rows
            for j = 1:cols
               if i < radius+1 || i > rows-radius || ... % Edge effects, need to truncate
                     j < radius+1 || j > cols-radius     % filter window
                  rS = max(1,i-radius); % rowStart
                  rE = min(rows, i+radius); % rowEnd
                  cS = max(1, j-radius); % colStart
                  cE = min(cols, j+radius); % colEnd
                  samp = bsxfun(@rdivide, sum(dLdy(:,:,rS:rE,cS:cE), 1), ...
                                          obj.edgeFactor(:,:,rS:rE,cS:cE));
                  topRoom = min(radius, i-1);
                  bottomRoom = min(radius, rows-i);
                  leftRoom = min(radius, j-1);
                  rightRoom = min(radius, cols-j);
                  center = radius+1;
                  truncW = obj.W(1,1,center-topRoom:center+bottomRoom, ...
                                     center-leftRoom:center+rightRoom);
                  convTerm(:,:,i,j) = sum(sum(bsxfun(@times, truncW, samp), 4), 3);
               else   % Interior, can use full filter (still need to account for edgeFactors)
                  rS = i-radius;
                  rE = i+radius;
                  cS = j-radius;
                  cE = j+radius;
                  samp = bsxfun(@rdivide, sum(dLdy(:,:,rS:rE, cS:cE), 1), ...
                                          obj.edgeFactor(:,:,rS:rE,cS:cE));
                  convTerm(:,:,i,j) = sum(sum(bsxfun(@times, obj.W, samp), 4), 3);
               end
            end
         end
         dLdx = bsxfun(@minus, dLdy, convTerm);
      end
      
      function W = compute_weights(obj, channels)
         % Assumes windowSize is odd
         x = obj.gpuState.linspace(-2, 2, obj.windowSize);
         W = exp(-bsxfun(@plus, x.*x, x'.*x')/2)/(2*pi);
         W = shiftdim(W/(channels*sum(W(:))), -2); % 1 x 1 x windowSize x windowSize
      end
      
   end
end

