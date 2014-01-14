classdef Conv2DContrastNormalization < matlab.mixin.Copyable
   % Performs subtractive and divisive contrast normalization over Gaussian
   % window and accross feature maps (see LeCun2010 "Convolutional networks
   % and applications in vision")
   
   properties
      windowSize % (wS) The Gaussian weighting window is a wS x wS square
      W % weight coefficients for Gaussian weighting window
      gpuState
   end
   
   methods
      function obj = Conv2DContrastNormalization(windowSize, gpu)
         obj.windowSize = windowSize;
         if nargin < 2
            obj.gpuState = GPUState();
         else
            obj.gpuState = GPUState(gpu);
         end
         obj.W = obj.compute_weights();
      end
      
      function y = feed_forward(obj, x, isSave)
         radius = (obj.windowSize-1)/2;
         [C, N, rows, cols] = size(x);
         v = obj.gpuState.nan(size(x));
         
         % Apply subtractive normalization
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
                  truncW = obj.W(center-topRoom:center+bottomRoom, ...
                                 center-leftRoom:center+rightRoom);
                  truncW = truncW/sum(truncW(:));
                  v(:,:,i,j) = bsxfun(@minus, x(:,:,i,j), ...
                     sum(sum(sum(bsxfun(@times, truncW, samp), 4), 3, 1)));
               else   % Interior, can use full filter
                  samp = x(:,:,i-radius:i+radius, j-radius:j+radius);
                  v(:,:,i,j) = bsxfun(@minus, x(:,:,i,j), ...
                     sum(sum(sum(bsxfun(@times, obj.W, samp), 4), 3), 1));
               end
            end
         end
         
         % Apply divisive normalization
         sigma = obj.gpuState.nan(1,N,rows,cols);
         for i = 1:rows
            for j = 1:cols
               if i < radius+1 || i > rows-radius || ... % Edge effects, need to truncate
                     j < radius+1 || j > cols-radius     % filter window
                  samp = v(:,:,max(1,i-radius):min(rows,i+radius), ...
                               max(1,j-radius):min(cols,j+radius));
                  topRoom = min(radius, i-1);
                  bottomRoom = min(radius, rows-i);
                  leftRoom = min(radius, j-1);
                  rightRoom = min(radius, cols-j);
                  center = radius+1;
                  truncW = obj.W(center-topRoom:center+bottomRoom, ...
                                 center-leftRoom:center+rightRoom);
                  truncW = truncW/sum(truncW(:));
                  sigma(1,:,i,j) = sqrt(sum(sum(sum(bsxfun(@times, truncW, samp.*samp), ...
                                         4), 3), 1));
               else   % Interior, can use full filter
                  samp = v(:,:,i-radius:i+radius, j-radius:j+radius);
                  sigma(1,:,i,j) = sqrt(sum(sum(sum(bsxfun(@times, obj.W, samp.*samp), ...
                                          4), 3), 1));
               end
            end
         end
         meanSigma = mean(mean(sigma, 4), 3);
         y = bsxfun(@rdivide, v, max(sigma, meanSigma));
      end
      
      function dLdx = backprop(obj, dLdy)
         
      end
      
      function W = compute_weights(obj)
         % Assumes windowSize is odd
         x = obj.gpuState.linspace(-2, 2, obj.windowSize);
         W = exp(-bsxfun(@plus, x.*x, x'.*x')/2)/(2*pi);
         W = W/sum(W(:));
      end
      
   end
end

