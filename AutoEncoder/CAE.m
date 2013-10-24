classdef CAE < AutoEncoder
   % Contractive AutoEncoder
   
   properties     
      JacCoeff
      HessCoeff
      HessNoise
      HessBatchSize
   end
   
   methods
      function obj = CAE(varargin)
         obj = obj@AutoEncoder(varargin{:});
         p = inputParser;
         p.KeepUnmatched = true;
         p.addParamValue('JacCoeff', []);
         p.addParamValue('HessCoeff', []);
         p.addParamValue('HessNoise', .1);
         p.addParamValue('HessBatchSize', 20);
         parse(p, varargin{:});
         
         obj.JacCoeff = p.Results.JacCoeff;
         obj.HessCoeff = p.Results.HessCoeff;
         obj.HessNoise = p.Results.HessNoise;
         obj.HessBatchSize = p.Results.HessBatchSize;
      end
      
      function [grad, xRecon] = gradient(obj, xIn, xTarget, ~)
         xIn(isnan(xIn)) = 0;
         xCode = obj.encodeLayer.feed_forward(xIn);
         [decodeGrad, dLdxCode, xRecon] = obj.decodeLayer.backprop(xCode, xTarget);
         [encodeGrad, ~, dydz] = obj.encodeLayer.backprop(xIn, xCode, dLdxCode);
         penalty = obj.compute_contraction_penalty_gradient(xIn, dydz);
         encodeGrad = cellfun(@(grad, pen) grad + pen, encodeGrad, penalty, ...
                                 'UniformOutput', false);
         
         if obj.isTiedWeights
            if ndims(encodeGrad{1}) <= 2
               grad = {encodeGrad{1}+decodeGrad{1}', encodeGrad{2}, decodeGrad{2}};
            else
               grad = {encodeGrad{1}+permute(decodeGrad{1}, [2, 1, 3]), ...
                        encodeGrad{2}, decodeGrad{2}};
            end
         else
            grad = [encodeGrad, decodeGrad];
         end
      end
      
      function penalty = compute_contraction_penalty_gradient(obj, xIn, dydz)
         dydzSqMean = mean(dydz.*dydz, 2); % L2 x 1 (x k)
         penalty = obj.JacCoeff*bsxfun(@times, obj.encodeLayer.params{1}, dydzSqMean);
         
         if ~isempty(obj.HessCoeff)
            [L1, N] = size(xIn);
            permvec = randperm(N, obj.HessBatchSize);
            dydz = dydz(:,permvec);
            xIn2 = xIn(:,premvec) + obj.HessNoise*obj.gpuState.randn([L1, obj.HessBatchSize]);
            xCode2 = obj.encodeLayer.feed_forward(xIn2);
            dydz2= obj.encodeLayer.compute_dydz(xIn2, xCode2);
            diff = dydz - dydz2;
            diffSqMean = mean(diff.*diff, 2);
            penalty = penalty + obj.HessCoeff*bsxfun(@times, obj.encodeLayer.params{1}, ...
                                                        diffSqMean);
         end    
      end
      
      function objCopy = copy(obj)
         objCopy = copy@AutoEncoder(obj);
         objCopy.JacCoeff = obj.JacCoeff;
         objCopy.HessCoeff = obj.HessCoeff;
         objCopy.HessNoise = obj.HessNoise;
         objCopy.HessBatchSize = obj.HessBatchSize;
      end
   end
   
end
