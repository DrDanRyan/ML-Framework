classdef PRBEPOutputLayer < StandardOutputLayer
% Precision-Recall Break Even Point SVM output layer. Uses a linear output
% layer with a nonlinear Loss function that acts on SETS of inupts and
% targets. Should be used in full batch mode.

   properties
      nonlinearity = @(x) x; % not actually used, need to define to inherit StandardOutputLayer
   end

   methods
      function obj = PRBEPOutputLayer(inputSize, varargin)
         obj = obj@StandardOutputLayer(inputSize, 1, varargin{:});
      end
      
      function [dLdz, y] = dLdz(obj, x, t)
         y = obj.feed_forward(x);
         yStar = obj.compute_yStar_and_Delta(y, t);
         dLdz = yStar - t;
      end
      
      function y = feed_forward(obj, x)
         % output layer is linear; redefine feedforward to save computation
         % vs. using:   nonlinearity = @(x) x;
         y = bsxfun(@plus, obj.params{1}*x, obj.params{2});
      end
      
      function loss = compute_loss(obj, y, t)
         [yStar, Delta] = obj.compute_yStar_and_Delta(y, t);
         loss = Delta + (yStar - t)*y';
      end
      
      function [yStar, Delta] = compute_yStar_and_Delta(obj, y, t)
         [~, sortIdx] = sort(y);
         posIdx = sortIdx(t==1);
         negIdx = sortIdx(t~=1);
         yPos = y(posIdx)';
         yNeg = y(negIdx)';
         nPos = length(posIdx); 
         nNeg = length(negIdx);    
         
         c = -flip(triu(obj.gpuState.ones(nPos+1,nPos)) , 2);
         a = flip(triu(obj.gpuState.ones(nPos+1,nPos)) , 1);
         yPrimePos = a+c;
         
         b = triu(obj.gpuState.ones(nPos+1, nNeg), nNeg - nPos);
         d = -tril(obj.gpuState.ones(nPos+1, nNeg), nNeg - nPos -1);
         yPrimeNeg = b+d;
         
         delta = 100*obj.gpuState.linspace(nPos, 0, nPos+1)'/nPos;
         
         [~, bestIdx] = max(delta + yPrimePos*yPos + yPrimeNeg*yNeg);
         Delta = delta(bestIdx);
         yStar = obj.gpuState.zeros(size(t));
         yStar(posIdx) = yPrimePos(bestIdx, :);
         yStar(negIdx) = yPrimeNeg(bestIdx, :);
         
%          for a = 0:nPos
%             b = nPos - a; % c = b at breakeven point
%             d = nNeg - b;
%             yPrime(posIdx(1:b)) = -1;
%             if b < nPos
%                yPrime(posIdx(b+1:end)) = 1;
%             end
%             yPrime(negIdx(1:d)) = -1;
%             if d < nNeg
%                yPrime(negIdx(d+1:end)) = 1;
%             end
%             DeltaPrime = 100*b/(a+b);
%             loss = DeltaPrime + yPrime*y';
%             if loss > maxLoss
%                yStar = yPrime;
%                Delta = DeltaPrime;
%                maxLoss = loss;
%             end
%          end
      end
      
   end
end

