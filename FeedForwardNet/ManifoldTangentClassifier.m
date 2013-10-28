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
         else
            mask = [];
         end
         
         % feed_forward through hiddenLayers
         [y, dydx] = obj.feed_forward(x, mask); % dydx are the layer Jacobians
         
         % get outputLayer output and backpropagate loss
         [grad, output, dLdx] = obj.backprop(x, y, t, mask);
         
         % compute mainfold tangent penalty gradient and add it to grad
         penalty = obj.compute_penalty_gradient(x, u, y, dydx, mask);
         penalty = obj.unroll_gradient(penalty);
         grad = cellfun(@plus, grad, penalty, 'UniformOutput', false);
      end
      
      function [y, dydx] = feed_forward(obj, x, mask)
         % feed_forward through hiddenLayers collecting Jacobians of each
         % layer mapping along the way
         
         nHiddenLayers = length(obj.hiddenLayers);
         y = cell(1, nHiddenLayers); % output from each hiddenLayer
         dydx = cell(1, nHiddenLayers); % the Jacobians of each layer mapping
         y{1} = obj.hiddenLayers{1}.feed_forward(x);
         dydz = obj.hiddenLayers{1}.compute_Dy(x, y{1}); % L2 x N (x k)
         dydx{1} = bsxfun(@times, dydz, obj.hiddenLayers{1}.params{1}); % L2 x L1
         if obj.isDropout
            y{1} = y{1}.*mask{2};
            dydx{1} = bsxfun(@times, dydx{1}, mask{2}');
            dydx{1} = bsxfun(@times, dydx{1}, mask{1});
         end

         for i = 2:nHiddenLayers
            y{i} = obj.hiddenLayers{i}.feed_forward(y{i-1});
            dydz = obj.hiddenLayers{i}.compute_Dy(y{i-1}, y{i});
            dydx{i} = bsxfun(@times, dydz, obj.hiddenLayers{i}.params{1})*dydx{i-1};
            if obj.isDropout
               y{i} = y{i}.*mask{i+1};
               dydx{i} = dydx{i}.*mask{i+1};
            end
         end
      end
      
      function penalty = compute_penalty_gradient(obj, x, u, y, dydx, mask)
         nHiddenLayers = length(obj.hiddenLayers);
         dodh = cell(1, nHiddenLayers);
         dhdx = cell(1, nHiddenLayers);
         
         % Compute forward and backwards cummulative products of Jacobians
         dodh{end} = dydx{end};
         dhdx{1} = dydx{1};
         for i = 2:nHiddenLayers
            revIdx = nHiddenLayers - i +1;
            dhdx{i} = dydx{i}*dhdx{i-1};
            dodh{revIdx} = dodh{revIdx+1}*dydx{revIdx};
         end
         
         
         
      end
   end   
end

