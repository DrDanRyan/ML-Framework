classdef AdaDelta < StepCalculator
   
   properties
      eps  % small constant used to fudge rms values
      tau  % time constant for moving averages
      
      gradSquared  % moving average of squared gradients
      stepSquared  % moving average of squared steps
   end
   
   methods
      function obj = AdaDelta(varargin)
         p = inputParser();
         p.addParamValue('eps', 1e-6);
         p.addParamValue('tau', .9);
         parse(p, varargin{:});
         obj.eps = p.Results.eps;
         obj.tau = p.Results.tau;
      end
      
      function take_step(obj, batch, model, ~)
         grad = model.gradient(batch);
         step = obj.compute_step(grad);
         model.increment_params(step);
      end
      
      function step = compute_step(obj, grad)
         if isempty(obj.gradSquared)
            obj.gradSquared = cellfun(@(grad) (1-obj.tau)*grad.*grad, grad, ...
                                      'UniformOutput', false);
            step = cellfun(@(gradAvg, grad) -sqrt(obj.eps)./(sqrt(gradAvg)+obj.eps).*grad, ...
                           obj.gradSquared, grad, 'UniformOutput', false);
            obj.stepSquared = cellfun(@(step) (1-obj.tau)*step.*step, step, ...
                                      'UniformOutput', false);                                       
         else
            obj.gradSquared = cellfun(@(avg, grad) obj.tau*avg + (1-obj.tau)*grad.*grad, ...
                                          obj.gradSquared, grad, 'UniformOutput', false);
            step = cellfun(@(stepAvg, gradAvg, grad) ...
                           -(sqrt(stepAvg)+obj.eps)./(sqrt(gradAvg)+obj.eps).*grad, ...
                           obj.stepSquared, obj.gradSquared, grad, 'UniformOutput', false);
            obj.stepSquared = cellfun(@(avg, step) obj.tau*avg + (1-obj.tau)*step.*step, ...
                                      obj.stepSquared, step, 'UniformOutput', false);            
         end
      end
      
      function reset(obj)
         obj.gradSquared = [];
         obj.stepSquared = [];
      end
      
   end
end

