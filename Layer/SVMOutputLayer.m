classdef SVMOutputLayer < StandardOutputLayer
   % A linear output layer with a hinge loss function raised to the power lossExponent.
   % Requires targets to take values {-1, 1} (as opposed to {0, 1} for LogisticOutputLayer)
   
   properties
      costRatio % multiplies the loss for incorrectly classifying positive (rarer) examples
      lossExponent % exponent of the hinge loss function (>= 1)
      nonlinearity % not used; Abstract property in StandardLayer
   end
   
   methods
      function obj = SVMOutputLayer(inputSize, varargin)
         obj = obj@StandardOutputLayer(inputSize, 1, varargin{:});
         p = inputParser();
         p.KeepUnmatched = true;
         p.addParamValue('lossExponent', 2, @(x) x > 1)
         p.addParamValue('costRatio', 1);
         parse(p, varargin{:});
         obj.lossExponent = p.Results.lossExponent;
         obj.costRatio = p.Results.costRatio;
      end
         
      function [dLdz, y] = compute_dLdz(obj, x, t)
         y = obj.feed_forward(x);
         if obj.costRatio == 1
            dLdz = -obj.lossExponent*t.*(max(1 - y.*t, 0).^(obj.lossExponent-1));
         else % obj.costRatio ~= 1
            posIdx = t==1;
            negIdx = t~=1;
            tPos = t(posIdx);
            yPos = y(posIdx);
            tNeg = t(negIdx);
            yNeg = y(negIdx);
            dLdz = obj.gpuState.zeros(size(y));
            dLdz(:,posIdx) = -obj.lossExponent*obj.costRatio*...
                                 tPos.*(max(1 - yPos.*tPos, 0).^(obj.lossExponent-1));
            dLdz(:,negIdx) = -obj.lossExponent*tNeg.*(max(1 - yNeg.*tNeg, 0).^(obj.lossExponent-1));
         end
      end
      
      function value = compute_dydz(obj, ~, y)
         value = obj.gpuState.ones(size(y));
      end
      
      function y = feed_forward(obj, x)
         % output layer is linear; redefine feedforward to save computation
         % vs. using:   nonlinearity = @(x) x;
         y = bsxfun(@plus, obj.params{1}*x, obj.params{2});
      end
      
      function loss = compute_loss(obj, y, t)
         if obj.costRatio == 1
            loss = mean(max(1 - y.*t, 0).^obj.lossExponent);
         else % costRatio ~= 1
            posIdx = t==1;
            negIdx = t~=1;
            tPos = t(posIdx);
            yPos = y(posIdx);
            
            tNeg = t(negIdx);
            yNeg = y(negIdx);
            
            loss = mean([obj.costRatio*max(1 - yPos.*tPos, 0).^obj.lossExponent, ...
                           max(1 - yNeg.*tNeg, 0).^obj.lossExponent]);
         end
      end
   end
   
end