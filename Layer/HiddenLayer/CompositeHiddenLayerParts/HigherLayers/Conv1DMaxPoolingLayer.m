classdef Conv1DMaxPoolingLayer < matlab.mixin.Copyable
   
   properties
      inputSize
      poolSize
      winners
   end
   
   methods
      function obj = Conv1DMaxPoolingLayer(poolSize)
         obj.poolSize = poolSize;
      end
      
      function y = feed_forward(obj, x, isSave)
         [nF, N, obj.inputSize] = size(x);
         remainder = mod(obj.inputSize, obj.poolSize);
         if remainder > 0
            if isa(x, 'gpuArray')
               padding = gpuArray.nan([nF, N, obj.poolSize - remainder], 'single');
            else
               padding = nan([nF, N, obj.poolSize - remainder]);
            end
            x = cat(3, x, padding);
         end
         x = reshape(x, nF, N, obj.poolSize, []);
         [y, idx] = max(x, [], 3);
         if nargin == 3 && isSave
            if isa(x, 'gpuArray')
               obj.winners = gpuArray.false(size(x));
            else
               obj.winners = false(size(x));
            end
            
            for i = 1:obj.poolSize
               obj.winners(:,:,i,:) = idx==i;
            end
         end
         y = permute(y, [1, 2, 4, 3]);
      end
      
      function dLdx = backprop(obj, dLdy)
         [nF, N, ~] = size(dLdy);
         dLdx = bsxfun(@times, obj.winners, permute(dLdy, [1,2,4,3]));
         obj.winners = [];
         dLdx = reshape(dLdx, nF, N, []);
         dLdx = dLdx(:,:,1:obj.inputSize);         
      end
      
   end
end

