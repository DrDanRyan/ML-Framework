classdef NesterovMomentum < StepCalculator
   % Uses Nesterov's Accelerated Gradient method to calculate next step.
   % Requires corresponding ParameterSchedule to provide 
   % params = {learnRate, momentum}.
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
         if ~isempty(obj.velocity)
            model.increment_params(obj.compute_first_step(params));
         end
         
         grad = model.gradient(batch);
         model.increment_params(obj.compute_second_step(params, grad));
      end
      
      function step = compute_first_step(obj, params)
         rho = params{2};
         if isscalar(rho)
            step = cellfun(@(vel) rho*vel, obj.velocity, 'UniformOutput', false);
         else
            step = cellfun(@(rho, vel) rho*vel, rho, obj.velocity, ...
                           'UniformOutput', false);
         end
      end
      
      function step = compute_second_step(obj, params, grad)
         [lr, rho] = params{:};
         if isempty(obj.velocity)
            if isscalar(lr)
               obj.velocity = cellfun(@(grad) -lr*grad, grad, ...
                                      'UniformOutput', false);
            else
               obj.velocity = cellfun(@(lr, grad) -lr*grad, lr, grad, ...
                                      'UniformOutput', false);
            end  
            step = obj.velocity;
         else
            if isscalar(lr)
               obj.velocity = cellfun(@(grad, vel) rho*vel - lr*grad, grad, ...
                                      obj.velocity, 'UniformOutput', false);
               step = cellfun(@(grad) -lr*grad, grad, 'UniformOutput', false);
            else
               obj.velocity = cellfun(@(lr, rho, grad, vel) rho*vel - lr*grad, ...
                                      lr, rho, grad, obj.velocity, ...
                                      'UniformOutput', false);
               step = cellfun(@(lr, grad) -lr*grad, lr, grad, ...
                                      'UniformOutput', false);               
            end            
         end         
      end
      
      function reset(obj)
         % Velocity is reset to empty
         obj.velocity = [];
      end
      
   end
end

