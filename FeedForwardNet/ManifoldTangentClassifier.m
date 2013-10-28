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
         % Need to modify for maxout networks
         nHiddenLayers = length(obj.hiddenLayers);
         
         % Compute dydz for each layer (including output)
         Dy = cell(1, nHiddenLayers+1);
         D2y = cell(1, nHiddenLayers+1);
         layer = obj.hiddenLayers{1};
         Dy{1} = obj.compute_Dy(x, y{1}, mask{2}, layer);
         if ~layer.isLocallyLinear
            D2y{1} = obj.compute_D2y(x, y{1}, mask{2}, layer);
         end         
         for i = 2:nHiddenLayers
            layer = obj.hiddenLayers{i};
            Dy{i} = obj.compute_Dy(y{i-1}, y{i}, mask{i+1}, layer);
            if ~layer.isLocallyLinear
               D2y{i} = obj.compute_D2y(y{i-1}, y{i}, mask{i+1}, layer);
            end
         end
         layer = obj.outputLayer;
         Dy{end} = obj.compute_Dy(y{end}, output, 1, layer);       
         if ~layer.isLocallyLinear
            D2y{end} = obj.compute_D2y(y{end}, output, 1, layer);
         end
         
         % Compute forward dhdx*u cumulative products
         dhdx_u = obj.compute_forward_Jacobian_u_products(u, Dy); % L2 x N x M (where M is nSingularVectors)
         
         % Compute backward Jacobian cumulative products
         dodh = obj.compute_backwards_Jacobian_products(Dy); % L2 x N x outputSize
         
         % Compute penalty terms
         penalty = cell(1, 2);
         dodx_u = shiftdim(permute(dhdx_u{end}, [2, 3, 1]), -1); % 1 x N x M x outputSize
         N = size(x, 2);
         
         % input layer
         layer = obj.hiddenLayers{1};
         L2 = layer.outputSize;
         temp1 = bsxfun(@times, reshape(dodh{1}, [L2, N, 1, outputSize]), dodx_u); % L2 x N x M x outputSize
         
         if ~layer.isLocallyLinear % Must include terms with D2y as a factor
            temp2 = bsxfun(@times, temp1, D2y{1});
            temp2 = bsxfun(@times, temp2, pagefun(@mtimes, layer.params{1}, u));
            prod = pagefun(@mtimes, temp2, x')/N; % L2 x L1 x M x outputSize
            sum1 = sum(sum(prod, 4), 3); % L2 x L1
            
            temp2 = mean(temp2, 2);
            penalty{2} = obj.tangentCoeff*sum(sum(temp2, 4), 3);
         else
            sum1 = 0;
            penalty{2} = 0;
         end
         
         temp2 = bsxfun(@times, temp1, Dy{1});
         prod = pagefun(@mtimes, temp2, permute(u, [2, 1, 3]))/N; % L2 x L1 x M x outputSize
         sum2 = sum(sum(prod, 4), 3);
         penalty{1} = obj.tangentCoeff*(sum1 + sum2);
      end
      
      function Dy = compute_Dy(obj, x, y, mask, layer)
         Dy = layer.compute_Dy(obj, x, y);
         if obj.isDropout
            Dy = Dy.*mask;
         end
      end
      
      function D2y = compute_D2y(obj, x, y, mask, layer)
         if layer.isLocallyLinear
            D2y = 0;
            return
         end
         
         D2y = layer.compute_D2y(obj, x, y);
         if obj.isDropout
            D2y = D2y.*mask;
         end
      end
      
      function dhdx_u = compute_forward_Jacobian_u_products(obj, u, dydz)
         % Need to modify for maxout networks (sum over k dimension after
         % dydz b* dhdx_u)
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
      
      function dodh = compute_backwards_Jacobian_products(obj, dydz)
         % Need to modify for maxout networks
         nHiddenLayers = length(obj.hiddenLayers);
         dodh = cell(1, nHiddenLayers);
         dodh{end} = bsxfun(@times, dydz{end}', outputLayer.params{1}');
         for i = nHiddenLayers-1:-1:1
            dodh{i} = bsxfun(@times, dydz{i}', dodh{i+1});
            dodh{i} = pagefun(@mtimes, obj.hiddenLayers{i}.params{1}', dodh{i});
         end
      end
      
      
   end   
end

