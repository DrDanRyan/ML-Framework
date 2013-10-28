classdef ManifoldTangentClassifier < FeedForwardNet
   
   properties
      nSingularVectors  % number of singular vectors stored per training example
      tangentCoeff      % coefficient for tangent penalty term
   end
   
   methods
      function obj = ManifoldTangentClassifier(varargin)
         obj = obj@FeedForwardNet(varargin{:});
      end
      
      function [grad, output, dLdx] = gradient(obj, batch)
         % Computes the gradient for batch input x and target t for all parameters in
         % each hiddenLayer and outputLayer. The 3D array, u, stores
         % singular vectors describing the data manifold near each training
         % exammple.
         
         x = batch{1};
         u = batch{2};
         t = batch{3};
         
         if obj.isDropout
            mask = obj.dropout_mask(x);
            x = x.*mask{1};
            u = u.*mask{1}; % not really advised to use input dropout with MTC
         else
            mask = [];
         end
         
         % feed_forward through hiddenLayers
         y = obj.feed_forward(x, mask); % dydx are the layer Jacobians
         
         % get outputLayer output and backpropagate loss
         [grad, output, dLdx] = obj.backprop(x, y, t, mask);
         
         % compute mainfold tangent penalty gradient and add it to grad
         penalty = obj.compute_penalty_gradient(x, u, y, output, mask);
         penalty = obj.unroll_gradient(penalty);
         grad = cellfun(@plus, grad, penalty, 'UniformOutput', false);
      end
      
      function penalty = compute_penalty_gradient(obj, x, u, y, output, mask)
         nHiddenLayers = length(obj.hiddenLayers);
         
         % Compute dydz for each layer (including output)
         dydz = cell(1, nHiddenLayers+1);
         dydz{1} = obj.hiddenLayers{1}.compute_Dy(x, y{1});
         if obj.isDropout
            dydz{1} = dydz{1}.*mask{2};
         end
         for i = 2:nHiddenLayers
            dydz{i} = obj.hiddenLayers{i}.compute_Dy(y{i-1}, y{i});
            if obj.isDropout
               dydz{i} = dydz{i}.*mask{i+1};
            end
         end
         dydz{end} = obj.outputLayer.compute_Dy(y{end}, output);       
         
         % Compute forward dhdx*u cumulative products
         dhdx_u = obj.compute_forward_Jacobian_u_products(u, dydz);
         
         % Compute backward

      end
      
      function dhdx_u = compute_forward_Jacobian_u_products(obj, u, dydz)
         nHiddenLayers = length(obj.hiddenLayers);
         dhdx_u = cell(1, nHiddenLayers+1);
         dhdx_u{1} = pagefun(@mtimes, obj.hiddenLayers{1}.params{1}, u);
         dhdx_u{1} = bsxfun(@times, dydz{1}, dhdx_u{1});
         for i = 2:nHiddenLayers
            dhdx_u{i} = pagefun(@mtimes, obj.hiddenLayers{i}.params{1}, dhdx_u{i-1});
            dhdx_u{i} = bsxfun(@times, dydz{i}, dhdx_u{i});
         end
         dhdx_u{end} = pagefun(@mtimes, obj.outputLayer.params{1}, dhdx_u{end-1});
         dhdx_u{end} = bsxfun(@times, dydz{end}, dhdx_u{end});
      end
   end   
end

