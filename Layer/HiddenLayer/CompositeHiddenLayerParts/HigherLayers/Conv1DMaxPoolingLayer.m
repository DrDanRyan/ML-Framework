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
      
      function xPool = feed_forward(obj, x, isSave)
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
         xPool = max(x, [], 3);
         if nargin == 3 && isSave
            obj.winners = bsxfun(@eq, xPool, x);
         end
         xPool = permute(xPool, [1, 2, 4, 3]);
      end
      
      function dLdyUnpool = backprop(obj, dLdy)
         [nF, N, ~] = size(dLdy);
         dLdyUnpool = bsxfun(@times, obj.winners, permute(dLdy, [1,2,4,3]));
         obj.winners = [];
         dLdyUnpool = reshape(dLdyUnpool, nF, N, []);
         dLdyUnpool = dLdyUnpool(:,:,1:obj.inputSize);         
      end
      
   end
end

