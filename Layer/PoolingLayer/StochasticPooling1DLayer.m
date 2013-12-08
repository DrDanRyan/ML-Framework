classdef StochasticPooling1DLayer < PoolingLayer
   
   properties
      poolSize
      inputSize
      winners
   end
   
   methods
      function obj = StochasticPooling1DLayer(poolSize)
         obj.poolSize = poolSize;
      end
      
      function xPool = pool(obj, x, isSave)
         % if isSave is false, use linear combination, else sample based on
         % multinomial probabilities
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
         probs = bsxfun(@rdivide, x, nansum(x, 3));         
         if nargin == 3 && isSave % sample for pooled values
            obj.winners = multinomial_sample(probs, 3);
            xPool = permute(nansum(x.*obj.winners, 3), [1, 2, 4, 3]);
         else % weighted average for pooled values
            xPool = permute(nansum(x.*probs, 3), [1, 2, 4, 3]);
         end
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

