classdef SplitNetwork < handle
   
   properties
      bottomNet   % a FFN with MSE loss function
      topNet      % a FFN with loss function that matches the problem
      h1star      % the hidden layer values used by topNet as inputs
      h1          % the hidden layer activations computed by the bottomNet
      u           % the scaled multiplier values for (h1star - h1) = 0 constraint
      rho         % the penalty coefficient for the Augmented Lagrangian
      modelState  % 'top' or 'bottom'
   end
   
   methods
      function obj = SplitNetwork(bottomNet, topNet, rho)
         if nargin > 0
            obj.bottomNet = bottomNet;
            obj.topNet = topNet;
            obj.rho = rho;

            % also need to init h1star and u
         end
      end
      
      function [grad, output] = gradient(obj, x, t)
         % Used as a facade to implement Model interface
         switch obj.modelState
            case 'top'
               [grad, output] = obj.topGrad(t);
            case 'bottom'
               [grad, output] = obj.bottomGrad(x);
         end
      end
      
      function [grad, output] = bottomGrad(obj, x)
         % compute gradient of bottomNet parameters
         t = obj.u + obj.h1star;
         [grad, output] = obj.bottomNet.gradient(x, t);
         grad = cellfun(@(g) obj.rho*g, grad, 'UniformOutput', false);
      end
      
      function [grad, output] = topGrad(obj, t)
         % compute gradient of topNet parameters concatenated with gradient
         % wrt h1star
         [topNetGrad, output, dLdx] = obj.topNet.gradient(obj.h1star, t);
         mask = single(dLdx~=0);
         h1starGrad = dLdx + mask.*obj.rho.*(obj.h1star - obj.h1 + obj.u);
         grad = [topNetGrad, {h1starGrad}];
      end
      
      function update_u(obj)
         obj.u = obj.u + obj.h1star - obj.h1;
      end
      
      function set_h1(obj, x)
         obj.h1 = obj.bottomNet.output(x);
      end
      
      function increment_params(obj, delta)
         % a Model interface facade function
         switch obj.modelState
            case 'top'
               obj.increment_top_params(delta);
            case 'bottom'
               obj.increment_bottom_params(delta);
         end
      end
      
      function increment_bottom_params(obj, delta)
         obj.bottomNet.increment_params(delta);
      end
      
      function increment_top_params(obj, delta)
         obj.topNet.increment_params(delta(1:end-1));
         obj.h1star = obj.h1star + delta{end};
      end
      
      function loss = compute_loss(obj, y, t)
         % A model interface facade function
         switch obj.modelState
            case 'top'
               loss = obj.compute_top_loss(y, t);
            case 'bottom'
               loss = obj.compute_bottom_loss(y);
            case 'full'
               loss = obj.compute_full_loss(y, t);
         end
      end
      
      function loss = compute_full_loss(obj, y, t)
         loss = obj.topNet.compute_loss(y, t);
      end
      
      function loss = compute_bottom_loss(obj, y)
         loss = obj.rho*obj.bottomNet.compute_loss(y, obj.h1star + obj.u);
      end
      
      function loss = compute_top_loss(obj, y, t)
         loss = obj.topNet.compute_loss(y, t) + ...
                     obj.rho*obj.bottomNet.compute_loss(obj.h1, obj.h1star + obj.u);
      end
      
      function y = output(obj, x)
         % model interface facade function
         switch obj.modelState
            case 'top'
               y = obj.top_output();
            case 'bottom'
               y = obj.bottom_output(x);
            case 'full'
               y = obj.full_output(x);
         end
      end
      
      function y = full_output(obj, x)
         y = obj.bottomNet.output(x);
         y = obj.topNet.output(y);
      end
      
      function y = bottom_output(obj, x)
         y = obj.bottomNet.output(x);
      end
      
      function y = top_output(obj)
         y = obj.topNet.output(obj.h1star);
      end
      
      function reset(obj)
         obj.bottomNet.reset();
         obj.topNet.reset();
      end
      
      function objCopy = copy(obj)
         objCopy = SplitNetwork();
         
         % Handle properties
         objCopy.topNet = obj.topNet.copy();
         objCopy.bottomNet = obj.bottomNet.copy();
         
         % Value properties
         objCopy.h1 = obj.h1;
         objCopy.h1star = obj.h1star;
         objCopy.rho = obj.rho;
         objCopy.u = obj.u;
         objCopy.modelState = obj.modelState;
      end
   end
   
end
