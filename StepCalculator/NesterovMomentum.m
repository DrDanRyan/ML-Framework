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
      function take_step(obj, batch, model, params)
         [learnRate, momentum] = params{:}; % learnRate and momentum are each cell arrays
                                            % they can be different for each component of grad
         
         if isempty(obj.velocity)
            grad = model.gradient(batch);
            obj.velocity = cellfun(@(lr, grad) -lr*grad, learnRate, grad, 'UniformOutput', false);
         else
            modelCopy = model.copy();
            modelCopy.increment_params(cellfun(@(m, v) m*v, momentum, obj.velocity, ...
                                             'UniformOutput', false));
            grad = modelCopy.gradient(batch);
            clear modelCopy
            obj.velocity = cellfun(@(lr, m, grad, vel) m*vel - lr*grad, learnRate, momentum, ...
                                             grad, obj.velocity, 'UniformOutput', false);   
         end
         model.increment_params(obj.velocity);
      end
      
      function reset(obj)
         % Velocity is reset to empty
         obj.velocity = [];
      end
   end
   
end

