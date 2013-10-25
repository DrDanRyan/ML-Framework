classdef ManifoldTangentClassifier < FeedForwardNet
   
   properties
      nSingularVectors  % number of singular vectors stored per training example
      tangentCoeff      % coefficient for tangent penalty term
   end
   
   methods
      function obj = ManifoldTangentClassifier(varargin)
         obj = obj@FeedForwardNet(varargin{:});
      end
      
      function [grad, output, dLdx] = gradient(obj, x, t, u)
         % Computes the gradient for batch input x and target t for all parameters in
         % each hiddenLayer and outputLayer. The 3D array, u, stores
         % singular vectors describing the data manifold near each training
         % exammple.
         
         if obj.isDropout
            mask = obj.dropout_mask(x);
            x = x.*mask{1};
         else
            mask = [];
         end
         
         % feed_forward through hiddenLayers
         [y, dhdx] = obj.feed_forward(x, mask);
         
         % get outputLayer output and backpropagate loss
         [grad, output, dLdx] = obj.backprop(x, y, t, u, dhdx, mask);
      end
      
      function [y, dhdx] = feed_forward(obj, x, mask)
         % feed_forward through hiddenLayers
         nHiddenLayers = length(obj.hiddenLayers);
         y = cell(1, nHiddenLayers); % output from each hiddenLayer
         dhdx = cell(1, nHiddenLayers);
         y{1} = obj.hiddenLayers{1}.feed_forward(x);
         dydz = obj.hiddenLayers{1}.compute_dydx(x, y{1});
         dhdx{1} = bsxfun(@times, dydz, obj.hiddenLayers{1}.params{1});
         if obj.isDropout
            y{1} = y{1}.*mask{2};
            dhdx{1} = dhdx{1}.*mask{2};
         end

         for i = 2:nHiddenLayers
            y{i} = obj.hiddenLayers{i}.feed_forward(y{i-1});
            dydz = obj.hiddenLayers{i}.compute_dydz(y{i-1}, y{i});
            dhdx{i} = bsxfun(@times, dydz, obj.hiddenLayers{i}.params{1})*dhdx{i-1};
            if obj.isDropout
               y{i} = y{i}.*mask{i+1};
               dhdx{i} = dhdx{i}.*mask{i+1};
            end
         end
      end
      
      function [grad, output, dLdx] = backprop(obj, x, y, t, u, dhdx, mask)
         if isempty(obj.hiddenLayers)
            [grad, dLdx, output] = obj.outputLayer.backprop(x, t);
            return;
         end
         
         nHiddenLayers = length(obj.hiddenLayers);
         dLdy = cell(1, nHiddenLayers); % derivative of loss function wrt hiddenLayer output
         grad = cell(1, nHiddenLayers+1); % gradient of hiddenLayers and outputLayer (last idx)
         [grad{end}, dLdy{end}, output] = obj.outputLayer.backprop(y{end}, t);
                     
         if obj.isDropout
            dLdy{end} = dLdy{end}.*mask{end};
         end
         
         for i = nHiddenLayers:-1:2
            [grad{i}, dLdy{i-1}] = obj.hiddenLayers{i}.backprop(y{i-1}, y{i}, ...
               dLdy{i});
            if obj.isDropout
               dLdy{i-1} = dLdy{i-1}.*mask{i};
            end
         end
         [grad{1}, dLdx] = obj.hiddenLayers{1}.backprop(x, y{1}, dLdy{1});
         if obj.isDropout
            dLdx = dLdx.*mask{1};
         end
         grad = obj.unroll_gradient(grad);
      end
   end
   
end

