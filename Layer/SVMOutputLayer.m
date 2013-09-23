classdef SVMOutputLayer < StandardOutputLayer
   % A linear output layer with a hinge loss function raised to the power hingeExp.
   % Requires targets to take values {-1, 1} (as opposed to {0, 1} for LogisticOutputLayer)
   
   properties
      costRatio % multiplies the loss for incorrectly classifying positive (rarer) examples
      hingeExp % exponent of the hinge loss function (>= 1)
      nonlinearity = @(x) x; % not used; Abstract property in StandardLayer
   end
   
   methods
      function obj = SVMOutputLayer(inputSize, varargin)
         obj = obj@StandardOutputLayer(inputSize, 1, varargin{:});
         p = inputParser();
         p.KeepUnmatched = true;
         p.addParamValue('hingeExp', 2, @(x) x >= 1)
         p.addParamValue('costRatio', 1);
         parse(p, varargin{:});
         obj.hingeExp = p.Results.hingeExp;
      end
         
      function [dLdz, y] = dLdz(obj, x, t)
         y = obj.feed_forward(x);
         if obj.costRatio == 1
            if obj.hingeExp > 1
               dLdz = -obj.hingeExp*t.*(max(1 - y.*t, 0).^(obj.hingeExp-1));
            else % hingeExp == 1
               dLdz = -t.*obj.gpuState.make_numeric(y.*t < 1);
            end
         else % obj.costRatio ~= 1
            posIdx = t==1;
            negIdx = t~=1;
            tPos = t(posIdx);
            yPos = y(posIdx);
            tNeg = t(negIdx);
            yNeg = y(negIdx);
            if obj.hingeExp > 1
               dLdz = -obj.hingeExp*(obj.costRatio*tPos.*(max(1 - yPos.*tPos, 0).^(obj.hingeExp-1)) ...
                         + tNeg.*(max(1 - yNeg.*tNeg, 0).^(obj.hingeExp-1)));
            else % hingeExp == 1
               dLdz = -obj.costRatio*tPos.*obj.gpuState.make_numeric(yPos.*tPos < 1) ...
                        -tNeg.*obj.gpuState.make_numeric(yNeg.*tNeg < 1);
            end
         end
      end
      
      function y = feed_forward(obj, x)
         % output layer is linear; redefine feedforward to save computation
         % vs. using:   nonlinearity = @(x) x;
         y = bsxfun(@plus, obj.params{1}*x, obj.params{2});
      end
      
      function loss = compute_loss(obj, y, t)
         if obj.costRatio == 1
            loss = mean(max(1 - y.*t, 0).^obj.hingeExp);
         else
            posIdx = t==1;
            negIdx = t~=1;
            tPos = t(posIdx);
            yPos = y(posIdx);
            
            tNeg = t(negIdx);
            yNeg = y(negIdx);
            
            loss = mean(obj.costRatio*max(1 - yPos.*tPos, 0).^obj.hingeExp + ...
                           max(1 - yNeg.*tNeg, 0).^obj.hingeExp);
         end
      end
   end
   
end