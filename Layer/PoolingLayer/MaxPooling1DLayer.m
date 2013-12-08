classdef MaxPooling1DLayer < PoolingLayer
   
   properties
      inputSize
      poolSize
      winners
   end
   
   methods
      function obj = MaxPooling1DLayer(poolSize)
         obj.poolSize = poolSize;
      end
      
      function xPool = pool(obj, x, isSave)
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
      
      function yUnpool = unpool(obj, y)
         [nF, N, ~] = size(y);
         yUnpool = bsxfun(@times, obj.winners, permute(y, [1,2,4,3]));
         obj.winners = [];
         yUnpool = reshape(yUnpool, nF, N, []);
         yUnpool = yUnpool(:,:,1:obj.inputSize);         
      end
   end
end

