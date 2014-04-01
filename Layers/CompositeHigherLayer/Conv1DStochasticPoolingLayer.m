classdef Conv1DStochasticPoolingLayer < CompositeHigherLayer & ...
                                        matlab.mixin.Copyable
   % Applies stochastic pooling to nonoverlapping regions of every channel of a
   % 1D input signal. Requires that the signal only takes nonnegative values.
   
   properties
      poolSize
      inputSize
      winners
   end
   
   methods
      function obj = Conv1DStochasticPoolingLayer(poolSize)
         obj.poolSize = poolSize;
      end
      
      function y = feed_forward(obj, x, isSave)
         % if isSave is false, use linear combination, else sample based on
         % multinomial probabilities
         [nF, N, obj.inputSize] = size(x);
         remainder = mod(obj.inputSize, obj.poolSize);
         if remainder > 0
            if isa(x, 'gpuArray')
               padding = ...
                  gpuArray.zeros([nF, N, obj.poolSize - remainder], 'single');
            else
               padding = zeros([nF, N, obj.poolSize - remainder]);
            end
            x = cat(3, x, padding);
         end
         x = reshape(x, nF, N, obj.poolSize, []);
         probs = bsxfun(@rdivide, x, sum(x, 3));         
         if nargin == 3 && isSave % sample for pooled values
            sample = multinomial_sample(probs, 3);
            y = permute(sum(x.*sample, 3), [1, 2, 4, 3]);
            obj.winners = logical(sample);
         else % weighted average for pooled values
            y = permute(sum(x.*probs, 3), [1, 2, 4, 3]);
         end
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

