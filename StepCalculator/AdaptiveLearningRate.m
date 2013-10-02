classdef AdaptiveLearningRate < StepCalculator
   % Implements Tom Schaul and Yann LeCun's "Adaptive learning rates for
   % stochastic, sparse, non-smooth gradients" (2013 arXiv) scheme.
   % Requires that the model is of type AdaptiveFeedForwardNet which uses
   % HiddenLayer and OutputLayer objects that respond to backprop_with_variance. 
   
   properties
      eps % small value used to prevent division by zero
      C % constant (>= 1) used to initially overestimate variance and diagonal hessian values
         % in order to slow down learning until estimates are accurate.
      n0 % number of samples used to initialize moving averages
         
      gradAvg % a moving average of past gradient values
      gradVarAvg % a moving average of past gradVariance values
      hessAvg % an exponential moving average of finite difference estimates of the diagonal of the Hessian
      hessVarAvg % a moving average of finite difference estimates of the variance of Hessian diag
      
      learnRates % current per parameter learning rates
      memorySize % current per parameter memories for each parameters moving average
   end
   
   methods
      function obj = AdaptiveLearningRate(varargin)
         p = inputParser;
         p.addParamValue('eps', 1e-5);
         p.addParamValue('C', 5);
         p.addParamValue('n0', 10);
         parse(p, varargin{:});
         
         obj.eps = p.Results.eps;
         obj.C = p.Results.C;
      end
      
      function take_step(obj, x, t, model, ~)
         if isempty(obj.gradAvg)
            N = size(x, 2);
            idx = randsample(N, obj.n0);
            obj.initialize_averages(x(:,idx), t(:,idx), model)
         end
            
         
      end
      
      function initialize_averages(obj, x, t, model)
         gra
      end
      
      function reset(obj)
         obj.gradAvg = [];
         obj.gradVarAvg = [];
         obj.hessAvg = [];
         obj.hessVarAvg = [];
         obj.learnRates = [];
         obj.memorySize = [];
      end
   end
   
end

