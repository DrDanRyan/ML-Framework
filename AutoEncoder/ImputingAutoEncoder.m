classdef ImputingAutoEncoder < DAE
   % Learns missing values by treating them as extra parameters to be
   % learned by backprop. Assumes data has been centered to zero mean.
   % Imposes L2 regularization penalty on learned values to prevent extreme
   % values (like a prior enforcing the observed mean).
   
   properties
      isImputeValues % boolean: true => NaN vals imputed prior to updating model
                     %          false => NaN vals are ignored (sub-model used)
      lam % coefficient for sum of squares regularization on imputed data
      imputedData % imputed values for last batch processed, N x 1 vecotr
                  % where N is number of NaN values in last batch 
      nSteps % number of steps to take for imputing data
      stepCalculator % NesterovMomentum StepCalculator
      lr % learning rate for data imputation
      rho % momentum for data imputation
   end
   
   methods
      function obj = ImputingAutoEncoder(varargin)
         obj = obj@DAE(varargin{:});
         p = inputParser();
         p.KeepUnmatched = true;
         p.addParamValue('isImputeValues', false);
         p.addParamValue('lam', .1);
         p.addParamValue('nSteps', 30);
         p.addParamValue('lr', .005);
         p.addParamValue('rho', .6);
         parse(p, varargin{:});
         
         obj.isImputeValues = p.Results.isImputeValues;
         obj.lam = p.Results.lam;
         obj.nSteps = p.Results.nSteps;
         obj.stepCalculator = NesterovMomentum();
         obj.lr = p.Results.lr;
         obj.rho = p.Results.rho;
      end
      
      function [grad, xRecon] = gradient(obj, batch)
         if obj.isImputeValues
            x = obj.impute_values(batch);
            [grad, xRecon] = gradient@DAE(obj, {x});
         else
            [grad, xRecon] = gradient@DAE(obj, batch);
         end       
      end
            
      function x = impute_values(obj, batch)
         x = batch{1};
         isNaN = batch{2};
         params = {obj.lr, obj.rho};
         obj.stepCalculator.reset();
         
         for i = 1:obj.nSteps
            if i > 1
               step = obj.stepCalculator.compute_first_step(params);
               x(isNaN) = x(isNaN) + step{1};
            end
            
            h = obj.encodeLayer.feed_forward(x, true);
            [~, dLdh, xRecon] = obj.decodeLayer.backprop(h, x);
            [~, dLdx] = obj.encodeLayer.backprop(x, h, dLdh);
            grad = {dLdx(isNaN) + (1 + obj.lam)*x(isNaN) - xRecon(isNaN)};
            step = obj.stepCalculator.compute_second_step(params, grad);
            x(isNaN) = x(isNaN) + step{1};
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
         objCopy.isImputeValues = obj.isImputeValues;
         objCopy.gpuState = obj.gpuState;
         objCopy.noiseType = obj.noiseType;
         objCopy.noiseLevel = obj.noiseLevel;
         objCopy.lam = obj.lam;
         objCopy.imputedData = obj.imputedData;
         objCopy.nSteps = obj.nSteps;
         objCopy.lr = obj.lr;
         objCopy.rho = obj.rho;
      end
   end
   
end

