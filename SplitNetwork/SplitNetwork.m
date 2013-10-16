classdef SplitNetwork < handle
   
   properties
      bottomLayer % an outputLayer with MSE loss function
      topLayer    % an outputLayer that matches the problem loss function
      h1          % the hidden layer values computed by bottomLayer
      h1star      % the hidden layer values used by topLayer
      u           % the scaled multiplier values for (h1star - h1) = 0 constraint
      rho         % the penalty coefficient for the Augmented Lagrangian
      modelState  % 'top' or 'bottom'
   end
   
   methods
      function obj = SplitNetwork(bottomLayer, topLayer, rho)
         if nargin > 0
            obj.bottomLayer = bottomLayer;
            obj.topLayer = topLayer;
            obj.rho = rho;

            % also need to init h1star and u
            % if beginning with topLayer update, need to compute h1
         end
      end
      
      function grad = gradient(obj, x, t)
         % Used as a facade to implement Model interface
         switch obj.modelState
            case 'top'
               grad = obj.topGrad(t);
            case 'bottom'
               grad = obj.bottomGrad(x);
         end
      end
      
      function grad = bottomGrad(obj, x)
         % compute gradient of bottomLayer parameters
         t = obj.u + obj.h1star;
         grad = obj.bottomLayer.backprop(x, t);
         grad = cellfun(@(g) obj.rho*g, grad, 'UniformOutput', false);
      end
      
      function grad = topGrad(obj, t)
         % compute gradient of topLayer parameters concatenated with gradient
         % wrt h1star
         [topLayerGrad, dLdx] = obj.topLayer.backprop(obj.h1star, t);
         h1starGrad = dLdx + obj.rho*(obj.h1star - obj.h1 + obj.u);
         grad = [topLayerGrad, {h1starGrad}];
      end
      
      function update_u(obj)
         obj.u = obj.u + obj.h1star - obj.h1;
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
         obj.bottomLayer.increment_params(delta);
      end
      
      function increment_top_params(obj, delta)
         obj.topLayer.increment_params(delta(1:2));
         obj.h1star = obj.h1star + delta{3};
      end
      
      function loss = compute_loss(obj, y, t)
         % A model interface facade function
         switch obj.modelState
            case 'top'
               loss = obj.compute_top_loss(y, t);
            case 'bottom'
               loss = obj.compute_bottom_loss(y);
         end
      end
      
      function loss = compute_bottom_loss(obj, y)
         obj.h1 = y;
         loss = obj.rho*obj.bottomLayer.compute_loss(obj.h1, obj.h1star + obj.u);
      end
      
      function loss = compute_top_loss(obj, y, t)
         loss = obj.topLayer.compute_loss(y, t) + ...
                     obj.rho*obj.bottomLayer.compute_loss(obj.h1, obj.h1star + obj.u);
      end
      
      function y = output(obj, x)
         % model interface facade function
         switch obj.modelState
            case 'top'
               y = obj.top_output();
            case 'bottom'
               y = obj.bottom_output(x);
         end
      end
      
      function y = bottom_output(obj, x)
         y = obj.bottomLayer.feed_forward(x);
      end
      
      function y = top_output(obj)
         y = obj.topLayer.feed_forward(obj.h1star);
      end
      
      function reset(obj)
         obj.bottomLayer.init_params();
         obj.topLayer.init_params();
      end
      
      function objCopy = copy(obj)
         objCopy = SplitNetwork();
         objCopy.topLayer = obj.topLayer.copy();
         objCopy.bottomLayer = obj.bottomLayer.copy();
         objCopy.h1 = obj.h1;
         objCopy.h1star = obj.h1star;
         objCopy.rho = obj.rho;
         objCopy.u = obj.u;
         objCopy.modelState = obj.modelState;
      end
   end
   
end
