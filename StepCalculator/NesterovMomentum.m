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
         [lr, rho] = params{:}; % lr (learning rate) and rho (momentum) are either 
                                % scalars or cell arrays the same size as grad
         
         if isempty(obj.velocity)
            grad = model.gradient(batch);
            if isscalar(lr)
               obj.velocity = cellfun(@(grad) -lr*grad, grad, 'UniformOutput', false);
            else
               obj.velocity = cellfun(@(lr, grad) -lr*grad, lr, grad, 'UniformOutput', false);
            end
            model.increment_params(obj.velocity);
         else
            if isscalar(lr)
               model.increment_params(cellfun(@(vel) rho*vel, obj.velocity, ...
                                                 'UniformOutput', false));
               grad = model.gradient(batch);
               obj.velocity = cellfun(@(grad, vel) rho*vel - lr*grad, grad, ...
                                                obj.velocity, 'UniformOutput', false);
               model.increment_params(cellfun(@(grad) -lr*grad, grad, 'UniformOutput', false));
            else
               model.increment_params(cellfun(@(rho, vel) rho*vel, rho, obj.velocity, ...
                                                'UniformOutput', false));
               grad = model.gradient(batch);
               obj.velocity = cellfun(@(lr, rho, grad, vel) rho*vel - lr*grad, lr, rho, ...
                                                grad, obj.velocity, 'UniformOutput', false);
               model.increment_params(cellfun(@(lr, grad) -lr*grad, lr, grad, ...
                                                                  'UniformOutput', false));
            end
         end
      end
      
      function reset(obj)
         % Velocity is reset to empty
         obj.velocity = [];
      end
      
   end
end

