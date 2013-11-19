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
         [learnRate, momentum] = params{:}; % learnRate and momentum are either scalars or cell 
                                            % arrays the same size as grad
         
         if isempty(obj.velocity)
            grad = model.gradient(batch);
            if isscalar(learnRate)
               obj.velocity = cellfun(@(grad) -learnRate*grad, grad, 'UniformOutput', false);
            else
               obj.velocity = cellfun(@(lr, grad) -lr*grad, learnRate, grad, 'UniformOutput', false);
            end
         else
            modelCopy = model.copy();
            if isscalar(learnRate)
               modelCopy.increment_params(cellfun(@(v) momentum*v, obj.velocity, ...
                                                 'UniformOutput', false));
               grad = modelCopy.gradient(batch);
               clear modelCopy
               obj.velocity = cellfun(@(grad, vel) momentum*vel - learnRate*grad, grad, ...
                                                obj.velocity, 'UniformOutput', false);
            else
               modelCopy.increment_params(cellfun(@(m, v) m*v, momentum, obj.velocity, ...
                                                'UniformOutput', false));
               grad = modelCopy.gradient(batch);
               clear modelCopy
               obj.velocity = cellfun(@(lr, m, grad, vel) m*vel - lr*grad, learnRate, momentum, ...
                                                grad, obj.velocity, 'UniformOutput', false);   
            end
         end
         model.increment_params(obj.velocity);
      end
      
      function reset(obj)
         % Velocity is reset to empty
         obj.velocity = [];
      end
   end
   
end

