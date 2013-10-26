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
         p.addParamValue('JacCoeff', .1);
         p.addParamValue('HessCoeff', []);
         p.addParamValue('HessNoise', .1);
         p.addParamValue('HessBatchSize', 10);
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
         [encodeGrad, ~, Dy] = obj.encodeLayer.backprop(xIn, xCode, dLdxCode);
         penalty = obj.compute_contraction_penalty_gradient(xIn, xCode, Dy);
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
      
      function penalty = compute_contraction_penalty_gradient(obj, xIn, xCode, Dy)
         [L1, N] = size(xIn);
         penalty = cell(1, 2);
         W = obj.encodeLayer.params{1};
         
         % Penalty from L2 norm of Jacobian map
         W_RowL2 = sum(W.*W, 2);
         D2y = obj.encodeLayer.compute_D2y(xIn, xCode);
         Dy_D2y_product = Dy.*D2y;
         penalty{2} = obj.JacCoeff*W_RowL2.*mean(Dy_D2y_product, 2);
         
         if ndims(Dy) <= 2
            x_outer_DyD2y = Dy_D2y_product*xIn'/N;
         else % ndims == 3 => Maxout layer
            x_outer_DyD2y = pagefun(@mtimes, Dy_D2y_product, x')/N;
         end
         penalty{1} = obj.JacCoeff*(bsxfun(@times, W_RowL2, x_outer_DyD2y) + ...
                                    bsxfun(@times, W, mean(Dy.*Dy, 2)));
         

         % Penalty from approximation to L2 norm of Hessian map
         if ~isempty(obj.HessCoeff)
            xEps = bsxfun(@plus, xIn, ...
                           obj.HessNoise*obj.gpuState.randn([size(xIn), obj.HessBatchSize]));
            xEpsLong = reshape(xEps, L1, []);
            xCodeEps = obj.encodeLayer.feed_forward(xEpsLong);
            DyEps = obj.encodeLayer.compute_Dy(xEpsLong, xCodeEps);
            clear xEpsLong
            xCodeEps = reshape(xCodeEps, [L2, N, obj.HessBatchSize]);
            DyEps = reshape(DyEps, L2, N, obj.HessBatchSize, []);
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
