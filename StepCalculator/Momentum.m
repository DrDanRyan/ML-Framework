classdef Momentum < StepCalculator
   % A basic gradient descent with momentum StepCalculator. Requires that 
   % the owning GradientTrainer class' trainingSchedule has params = {learnRate, momentum}
   
   properties
      velocity
   end
   
   methods
      function take_step(obj, x, t, model, params)
         grad = model.gradient(x, t);
         [learnRate, momentum] = params{:};
         
         if isempty(obj.velocity)
            obj.velocity = cellfun(@(grad) learnRate*grad, grad, 'UniformOutput', false);
         else
            obj.velocity = cellfun(@(grad, vel) momentum*vel + learnRate*grad, ...
                                    grad, obj.velocity, 'UniformOutput', false);
         end
         model.increment_params(obj.velocity);
      end
      
      function reset(obj)
         obj.velocity = [];
      end
   end
   
end

