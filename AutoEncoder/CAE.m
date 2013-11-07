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
      
      function [grad, xRecon] = gradient(obj, batch)
         xTarget = batch{1};
         xIn = batch{1};
         xIn(isnan(xIn)) = 0;
         [xCode, zCode] = obj.encodeLayer.feed_forward(xIn);
         [decodeGrad, dLdxCode, xRecon] = obj.decodeLayer.backprop(xCode, xTarget);
         [encodeGrad, ~, Dy] = obj.encodeLayer.backprop(xIn, xCode, zCode, dLdxCode);
         penalty = obj.compute_contraction_penalty_gradient(xIn, xCode, Dy);
         encodeGrad = cellfun(@(grad, pen) grad + pen, encodeGrad, penalty, ...
                                 'UniformOutput', false);
         
         if obj.isTiedWeights
            if ndims(encodeGrad{1}) <= 2
               grad = {encodeGrad{1}+decodeGrad{1}', encodeGrad{2}, decodeGrad{2}};
            else % maxout layer
               grad = {encodeGrad{1}+permute(decodeGrad{1}, [2, 1, 3]), ...
                        encodeGrad{2}, decodeGrad{2}};
            end
         else
            grad = [encodeGrad, decodeGrad];
         end
      end
      
      function penalty = compute_contraction_penalty_gradient(obj, xIn, xCode, zCode, Dy)
         % NOTE: Currently, encodeLayer.isDiagonalDy = true is assumed
         [L1, N] = size(xIn);
         penalty = cell(1, 2);
         W = obj.encodeLayer.params{1};
         isLocallyLinear = obj.encodeLayer.isLocallyLinear;
         
         % Penalty from L2 norm of Jacobian map
         if isLocallyLinear
            penalty{2} = 0;
            penalty{1} = obj.JacCoeff*bsxfun(@times, W, mean(Dy.*Dy, 2));
         else
            W_RowL2 = sum(W.*W, 2);
            D2y = obj.encodeLayer.compute_D2y(zCode, xCode);
            Dy_D2y_product = Dy.*D2y;
            penalty{2} = obj.JacCoeff*W_RowL2.*mean(Dy_D2y_product, 2);
         
            if ndims(Dy) <= 2
               x_outer_DyD2y = Dy_D2y_product*xIn'/N;
            else % ndims == 3 => Maxout layer
               x_outer_DyD2y = pagefun(@mtimes, Dy_D2y_product, x')/N;
            end
            penalty{1} = obj.JacCoeff*(bsxfun(@times, W_RowL2, x_outer_DyD2y) + ...
                                       bsxfun(@times, W, mean(Dy.*Dy, 2)));
         end
         

         % Penalty from approximation to L2 norm of Hessian map
         if ~isempty(obj.HessCoeff)
            xIn = repmat(xIn, 1, obj.HessBatchSize);
            Dy = repmat(Dy, 1, obj.HessBatchSize);
            
            
            xEps = xIn + obj.HessNoise*obj.gpuState.randn([L1, N*obj.HessBatchSize]);
            xCodeEps = obj.encodeLayer.feed_forward(xEps);
            DyEps = obj.encodeLayer.compute_Dy(xEps, xCodeEps);
            
            if ~isLocallyLinear
               D2y = repmat(D2y, 1, obj.HessBatchSize);
               D2yEps = obj.encodeLayer.compute_D2y(xEps, xCodeEps);
            end
            clear xCodeEps
            
            hessPenalty = cell(1, 2);
            Dy_diff = Dy - DyEps;
            clear Dy DyEps
            
            if isLocallyLinear
               hessPenalty{2} = 0;
               hessPenalty{1} = obj.HessCoeff*bsxfun(@times, W, mean(Dy_diff.*Dy_diff, 2));
            else
               hessPenalty{2} = obj.HessCoeff*bsxfun(@times, W_RowL2, ...
                                                mean(Dy_diff.*(D2y - D2yEps), 2));
               if ndims(D2y) <= 2
                  temp1 = ((Dy_diff.*D2y)*xIn' - (Dy_diff.*D2yEps)*xEps')/(N*obj.HessBatchSize);
               else % Maxout layer
                  temp1 = (pagefun(@mtimes, Dy_diff.*D2y, xIn') - ...
                           pagefun(@mtimes, Dy_diff.*D2yEps, xEps'))/(N*obj.HessBatchSize);
               end
               hessPenalty{1} = obj.HessCoeff*(bsxfun(@times, W_RowL2, temp1) + ...
                                                bsxfun(@times, W, mean(Dy_diff.*Dy_diff, 2)));
            end
            
            penalty = cellfun(@plus, penalty, hessPenalty, 'UniformOutput', false);
         end    
      end
      
      function objCopy = copy(obj)
         objCopy = CAE;
         
         % Handle properties
         objCopy.encodeLayer = obj.encodeLayer.copy();
         objCopy.decodeLayer = obj.decodeLayer.copy();
         
         % Value properties
         objCopy.isTiedWeights = obj.isTiedWeights;
         objCopy.gpuState = obj.gpuState;
         objCopy.JacCoeff = obj.JacCoeff;
         objCopy.HessCoeff = obj.HessCoeff;
         objCopy.HessNoise = obj.HessNoise;
         objCopy.HessBatchSize = obj.HessBatchSize;
      end
   end
   
end
