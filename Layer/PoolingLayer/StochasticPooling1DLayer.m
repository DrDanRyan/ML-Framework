classdef StochasticPooling1DLayer < PoolingLayer
   % requires input (x) to be nonnegative values
   
   properties
      poolSize
      inputSize
      winners
   end
   
   methods
      function obj = StochasticPooling1DLayer(poolSize)
         obj.poolSize = poolSize;
      end
      
      function xPool = feed_forward(obj, x, isSave)
         % if isSave is false, use linear combination, else sample based on
         % multinomial probabilities
         [nF, N, obj.inputSize] = size(x);
         remainder = mod(obj.inputSize, obj.poolSize);
         if remainder > 0
            if isa(x, 'gpuArray')
               padding = gpuArray.zeros([nF, N, obj.poolSize - remainder], 'single');
            else
               padding = zeros([nF, N, obj.poolSize - remainder]);
            end
            x = cat(3, x, padding);
         end
         x = reshape(x, nF, N, obj.poolSize, []);
         probs = bsxfun(@rdivide, x, sum(x, 3));         
         if nargin == 3 && isSave % sample for pooled values
            sample = multinomial_sample(probs, 3);
            xPool = permute(sum(x.*sample, 3), [1, 2, 4, 3]);
            obj.winners = logical(sample);
         else % weighted average for pooled values
            xPool = permute(sum(x.*probs, 3), [1, 2, 4, 3]);
         end
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

