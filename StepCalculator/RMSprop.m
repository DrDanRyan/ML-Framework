classdef RMSprop < StepCalculator
   % rmsprop with NesterovMomentum
   
   properties
      eps % small constant used as a fudge to avoid division by zero
      tau % time constant for geometric moving average
      gradSquaredAvg % a geometric moving average of gradient squared
      velocity % used to implement Nesterov Momentum (using scaled gradients)
   end
   
   methods
      function obj = RMSprop(tau, eps)
         obj.tau = tau;
         if nargin < 2
            eps = 1e-8;
         end
         obj.eps = eps;
      end
      
      function take_step(obj, batch, model, params)
         [lr, rho] = params{:};
         if isempty(obj.velocity) % gradSquaredAvg should also be empty
            grad = model.gradient(batch);
            obj.gradSquaredAvg = cellfun(@(grad) grad.*grad, grad, 'UniformOutput', false);
            if isscalar(lr) % lr and rho are scalars
               obj.velocity = cellfun(@(grad, avg) -lr*grad./(sqrt(avg)+obj.eps), ...
                                          grad, obj.gradSquaredAvg, 'UniformOutput', false);
            else % lr and rho are cell arrays
               obj.velocity = cellfun(@(lr, grad, avg) -lr*grad./(sqrt(avg)+obj.eps), ...
                                      lr, grad, obj.gradSquaredAvg, 'UniformOutput', false);
            end
            model.increment_params(obj.velocity);
         else % velocity and gradSquaredAvg are not empty
            if isscalar(lr) % lr and rho are scalars
               model.increment_params(cellfun(@(vel) rho*vel, obj.velocity, ...
                                          'UniformOutput', false));
               grad = model.gradient(batch);
               obj.gradSquaredAvg = cellfun(@(avg, grad) obj.tau*avg + ...
                  (1-obj.tau)*grad.*grad, obj.gradSquaredAvg, grad, 'UniformOutput', false);
               obj.velocity = cellfun(@(vel, grad, avg) rho*vel - ...
                  lr*grad./(sqrt(avg)+obj.eps), obj.velocity, grad, obj.gradSquaredAvg, ...
                  'UniformOutput', false);
               model.increment_params(cellfun(@(grad, avg) -lr*grad./(sqrt(avg)+obj.eps), ...
                  grad, obj.gradSquaredAvg, 'UniformOutput', false));
            else % lr and rho are cell arrays
               model.increment_params(cellfun(@(rho, vel) rho*vel, rho, obj.velocity, ...
                  'UniformOutput', false));
               grad = model.gradient(batch);
               obj.gradSquaredAvg = cellfun(@(avg, grad) obj.tau*avg + ...
                  (1-obj.tau)*grad.*grad, obj.gradSquaredAvg, grad, 'UniformOutput', false);
               obj.velocity = cellfun(@(rho, vel, lr, grad, avg) rho*vel - ...
                  lr*grad./(sqrt(avg)+obj.eps), rho, obj.velocity, lr, grad, ...
                  obj.gradSquaredAvg, 'UniformOutput', false);
               model.increment_params(cellfun(@(lr, grad, avg) ...
                  -lr*grad./(sqrt(avg)+obj.eps), lr, grad, obj.gradSquaredAvg, ...
                  'UniformOutput', false));
            end
         end
      end
      
      function reset(obj)
         obj.gradSquaredAvg = [];
         obj.velocity = [];
      end
   end
   
end

