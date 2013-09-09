classdef NAG < StepCalculator
   % Uses Nesterov's Accelerated Gradient (NAG) method to calculate next
   % step.
   
   properties
      velocity
   end
   
   methods
      function take_step(obj, x, t, model, params)
         [learnRate, momentum] = params{:};
         
         if isempty(obj.velocity)
            grad = model.gradient(x, t);
            obj.velocity = cellfun(@(grad) learnRate*grad, grad, 'UniformOutput', false);
         else
            model.increment_params(obj.velocity);
            grad = model.gradient(x, t);
            model.increment_params(cellfun(@(v) -v, obj.velocity, 'UniformOutput', false));
            obj.velocity = cellfun(@(grad, vel) momentum*vel + learnRate*grad, grad, ...
                                       obj.velocity, 'UniformOutput', false);   
         end
         model.increment_params(obj.velocity);
      end
      
      function reset(obj)
         obj.velocity = [];
      end
   end
   
end

