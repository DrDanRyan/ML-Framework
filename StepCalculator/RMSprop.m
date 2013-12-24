classdef RMSprop < StepCalculator
   
   properties
      tau % time constant for geometric moving average
      gradSquaredAvg % a geometric moving average of gradient squared
   end
   
   methods
      function obj = RMSprop(tau)
         obj.tau = tau;
      end
      
      function take_step(obj, batch, model, params)
         lr = params{1};
         grad = model.gradient(batch);
         if isempty(obj.gradSquaredAvg)
            obj.gradSquaredAvg = cellfun(@(g) g.*g, grad, 'UniformOutput', false);
         else
            obj.gradSquaredAvg = cellfun(@(g, avg) obj.tau*avg + (1-obj.tau)*g.*g, ...
                                             grad, obj.gradSquaredAvg, 'UniformOutput', false);
         end
         step = cellfun(@(g, avg) -lr*g./(sqrt(avg)+1e-8), grad, obj.gradSquaredAvg, ...
                           'UniformOutput', false);
         model.increment_params(step);
      end
      
      function reset(obj)
         obj.gradSquaredAvg = [];
      end
   end
   
end

