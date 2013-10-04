classdef NesterovMomentum < StepCalculator
   % Uses Nesterov's Accelerated Gradient method to calculate next step.
   % Requires corresponding TrainingSchedule to use params = {learnRate, momentum}.
   %
   % Formula used for update:
   % look_ahead_gradient = model.gradient at current model params PLUS velocity
   % new_velocity = momentum*velocity - learnRate*look_ahead_gradient
   % step = new_velocity
   
   properties
      velocity
   end
   
   methods
      function take_step(obj, x, t, model, params)
         [learnRate, momentum] = params{:};
         
         if isempty(obj.velocity)
            grad = model.gradient(x, t);
            obj.velocity = cellfun(@(grad) -learnRate*grad, grad, 'UniformOutput', false);
         else
            modelCopy = model.copy();
            modelCopy.increment_params(obj.velocity);
            grad = model.gradient(x, t);
            clear modelCopy
            obj.velocity = cellfun(@(grad, vel) momentum*vel - learnRate*grad, grad, ...
                                       obj.velocity, 'UniformOutput', false);   
         end
         model.increment_params(obj.velocity);
      end
      
      function reset(obj)
         % Velocity is reset to empty
         obj.velocity = [];
      end
   end
   
end

