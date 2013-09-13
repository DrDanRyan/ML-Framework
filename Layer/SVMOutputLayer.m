classdef SVMOutputLayer < StandardOutputLayer
   % A linear output layer with a hinge loss function raised to the power hingeExp.
   % Requires targets to take values {-1, 1} (as opposed to {0, 1} for LogisticOutputLayer)
   
   properties
      hingeExp
      nonlinearity = @(x) x; % not used; Abstract property in StandardLayer
   end
   
   methods
      function obj = SVMOutputLayer(inputSize, varargin)
         obj = obj@StandardOutputLayer(inputSize, 1, varargin{:});
         p = inputParser();
         p.KeepUnmatched = true;
         p.addParamValue('hingeExp', 2, @(x) x >= 1)
         parse(p, varargin{:});
         obj.hingeExp = p.Results.hingeExp;
      end
         
      function [dLdz, y] = dLdz(obj, x, t)
         y = obj.feed_forward(x);
         if obj.hingeExp > 1
            dLdz = -obj.hingeExp*t.*(max(1 - y.*t, 0).^(obj.hingeExp-1));
         else % hingeExp == 1
            dLdz = -t.*obj.gpuState.make_numeric(y.*t < 1);
         end
      end
      
      function y = feed_forward(obj, x)
         % output layer is linear; redefine feedforward to save computation
         % vs. using:   nonlinearity = @(x) x;
         y = bsxfun(@plus, obj.params{1}*x, obj.params{2});
      end
      
      function loss = compute_loss(obj, y, t)
         loss = mean(max(1 - y.*t, 0).^obj.hingeExp);
      end
   end
   
end