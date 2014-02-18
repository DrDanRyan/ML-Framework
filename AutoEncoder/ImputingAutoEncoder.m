classdef ImputingAutoEncoder < DAE
   % Learns missing values by treating them as extra parameters to be
   % learned by backprop. Assumes data has been centered to zero mean.
   % Imposes L2 regularization penalty on learned values to prevent extreme
   % values (like a prior enforcing the observed mean).
   
   properties
      lam % coefficient for sum of squares regularization on imputed data
      imputedData % imputed values for last batch processed, N x 1 vecotr
                  % where N is number of NaN values in last batch
                  
      nSteps % number of steps to take for imputing data
      stepCalculator % currently must be AdaDelta 
                     % must have method: step = compute_step(obj, grad)
   end
   
   methods
      function obj = ImputingAutoEncoder(varargin)
         obj = obj@DAE(varargin{:});
         p = inputParser();
         p.KeepUnmatched = true;
         p.addParamValue('lam', .1);
         p.addParamValue('nSteps', 20);
         p.addParamValue('tau', .6);
         parse(p, varargin{:});

         obj.lam = p.Results.lam;
         obj.nSteps = p.Results.nSteps;
         obj.stepCalculator = AdaDelta('tau', p.Results.tau);
      end
      
      function [grad, xRecon] = gradient(obj, batch)
         x = obj.impute_values(batch);
         [grad, xRecon] = gradient@DAE(obj, {x});         
      end
            
      function x = impute_values(obj, batch)
         x = batch{1};
         isNaN = batch{2};
         obj.stepCalculator.reset();
         
         for i = 1:obj.nSteps
            h = obj.encodeLayer.feed_forward(x, true);
            [~, dLdh, xRecon] = obj.decodeLayer.backprop(h, x);
            [~, dLdx] = obj.encodeLayer.backprop(x, h, dLdh);
            grad = dLdx(isNaN) + (1 + obj.lam)*x(isNaN) - xRecon(isNaN);
            step = obj.stepCalculator.compute_step(grad);
            x(isNaN) = x(isNaN) + step;
         end
         
         obj.imputedData = x(isNaN);
      end
      
      function objCopy = copy(obj)
         objCopy = ImputingAutoEncoder();
         
         % Handle properties
         objCopy.encodeLayer = obj.encodeLayer.copy();
         objCopy.decodeLayer = obj.decodeLayer.copy();
         objCopy.stepCalculator = obj.stepCalculator.copy();
         
         % Value properties
         objCopy.isTiedWeights = obj.isTiedWeights;
         objCopy.gpuState = obj.gpuState;
         objCopy.noiseType = obj.noiseType;
         objCopy.noiseLevel = obj.noiseLevel;
         objCopy.lam = obj.lam;
         objCopy.imputedData = obj.imputedData;
         objCopy.nSteps = obj.nSteps;
      end
   end
   
end

