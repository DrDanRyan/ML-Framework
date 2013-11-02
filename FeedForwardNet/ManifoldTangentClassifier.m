classdef ManifoldTangentClassifier < FeedForwardNet
   
   properties
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
            u = u.*mask{1}; % not advised to use input dropout with MTC
                            % there would be different singular vectors for
                            % different (sub)nets ... ?
         else
            mask = [];
         end
         
         % feed_forward through hiddenLayers
         y = obj.feed_forward(x, mask);
         
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
         
         % Compute Dy and D2y for each layer and also cumulative forward
         % Jacobian*u products
         Dy = cell(1, nHiddenLayers+1);
         D2y = cell(1, nHiddenLayers+1);
         dhdx_u = cell(1, nHiddenLayers);
         
         layer = obj.hiddenLayers{1};
         [Dy{1}, D2y{1}] = obj.compute_Dy_values(x, y{1}, mask{2}, layer);
         dhdx_u{1} = obj.compute_dhdx_u(u, Dy{1}, layer);
         for i = 2:nHiddenLayers
            layer = obj.hiddenLayers{i};
            [Dy{i}, D2y{i}] = obj.compute_Dy_values(y{i-1}, y{i}, mask{i+1}, layer);
            dhdx_u{i} = obj.compute_dhdx_u(dhdx_u{i-1}, Dy{i}, layer);
         end
         layer = obj.outputLayer;
         [Dy{end}, D2y{end}] = obj.compute_Dy_values(y{end}, output, 1, layer);
         dodx_u = obj.compute_dhdx_u(dhdx_u{end}, Dy{end}, layer);
         dodx_u = shiftdim(permute(dodx_u, [2, 3, 1]), -1); % 1 x N x U x C
         
         % Compute backward Jacobian cumulative products
         dodh = obj.compute_backwards_Jacobian_products(Dy); % L2 x N x C
         
         % Compute penalty terms
         penalty = cell(1, nHiddenLayers+1);
         
         % input layer
         layer = obj.hiddenLayers{1};
         [penalty{1}{1}, penalty{1}{2}] = obj.compute_layer_penalty(x, y{1}, dodx_u, dodh{1}, ...
                                                                     u, Dy{1}, D2y{1}, layer);
                                                                  
         % other hiddenLayers                                                         
         for i = 2:nHiddenLayers
            layer = obj.hiddenLayers{i};
            [penalty{i}{1}, penalty{i}{2}] = obj.compute_hiddenLayer_penalty(y{i-1}, y{i}, dodx_u, dodh{i}, ...
                                                                     dhdx_u{i-1}, Dy{i}, D2y{i}, layer);
         end
         
         % outputLayer
         layer = obj.outputLayer;
         [penalty{end}{1}, penalty{end}{2}] = obj.compute_outputLayer_penalty(y{end}, output, dodx_u, 1, ...
                                                                     dhdx_u{nHiddenLayers}, Dy{end}, D2y{end}, layer);
         
      end
      
      function [Dy, D2y] = compute_Dy_values(obj, x, y, mask, layer)
         Dy = layer.compute_Dy(obj, x, y);
         if obj.isDropout
            if layer.isDiagonalDy
               Dy = Dy.*mask;
            else
               Dy = bsxfun(@times, Dy, shiftdim(mask', -1));
            end
         end 
         
         D2y = [];
         if ~layer.isLocallyLinear
            D2y = layer.compute_D2y(x, y, Dy);
            if obj.isDropout
               if layer.isDiagonalDy
                  D2y = D2y.*mask;
               else % D2y ~ z x z x N x y
                  D2y = bsxfun(@times, D2y, shiftdim(mask', -2));
               end
            end
         end
      end
      
      function [W_pen, b_pen] = compute_hiddenLayer_penalty(obj, x, y, dodx_u, dodh, ...
                                                      dhdx_u, Dy, D2y, layer)
                                                   
         % This is implemented under the assumption that layer.isDiagonalDy
         % = true. However, the outputLayer version removes this assumption.
         % Dimensions of the input:
         % x ~ L1 x N
         % y ~ L2 x N
         % dodx_u ~ 1 x N x U x C
         % dodh ~ L2 x N x C
         % dhdx_u ~ L1 x N x U
         % Dy ~ L2 x N
         % D2y ~ L2 x N         
         
         N = size(x, 2);
   
         temp1 = bsxfun(@times, permute(dodh, [1 2 4 3]), dodx_u); % L2 x N x U x C
         
         if ~layer.isLocallyLinear % Must include terms with D2y as a factor
            temp2 = bsxfun(@times, temp1, D2y);
            temp2 = bsxfun(@times, temp2, pagefun(@mtimes, layer.params{1}, dhdx_u));
            prod = pagefun(@mtimes, temp2, x')/N; % L2 x L1 x M x outputSize
            sum1 = sum(sum(prod, 4), 3); % L2 x L1
            
            temp2 = mean(temp2, 2);
            b_pen = obj.tangentCoeff*sum(sum(temp2, 4), 3);
         else
            sum1 = 0;
            b_pen = 0;
         end
         
         temp2 = bsxfun(@times, temp1, Dy);
         prod = pagefun(@mtimes, temp2, permute(dhdx_u, [2, 1, 3]))/N; % L2 x L1 x M x outputSize
         sum2 = sum(sum(prod, 4), 3);
         W_pen = obj.tangentCoeff*(sum1 + sum2);                                                                                                  
      end
      
      function [W_pen, b_pen] = compute_outputLayer_penalty(obj, x, y, dodx_u, dhdx_u, ...
                                                               Dy, D2y, layer)
         
      end
      
      function dhdx_u = compute_dhdx_u(~, prev_dhdx_u, Dy, layer)
         % prev_dhdx_u ~ L1 x N x U
         
         if ndims(layer.params{1}) == 2 
            W_times_prev = pagefun(@mtimes, layer.params{1}, prev_dhdx_u);
         else % maxout layer
            W_times_prev = pagefun(@mtimes, permute(layer.params{1}, [1, 2, 4, 3]), ...
                                       prev_dhdx_u); % L2 x N x U x k
         end
         
         if layer.isDiagonalDy
            if ndims(layer.params) == 2
               dhdx_u = bsxfun(@times, Dy, W_times_prev);
            else % maxout layer
               [L2, N, k] = size(layer.params{1});
               dhdx_u = bsxfun(@times, reshape(Dy, [L2, N, 1, k]), W_times_prev);
               dhdx_u = sum(dhdx_u, 4);
            end
         else % Dy ~ z x N x y (softmax layer or similar)
            [L2, N, U] = size(W_times_prev);
            dhdx_u = squeeze(pagefun(@mtimes, permute(Dy, [3, 1, 2]), ...
                           reshape(W_times_prev, [L2, 1, N, U])));  % probably can be optimized better
         end
      end
      
      function dodh = compute_dodh(obj, prev_dodh, Dy, layer)
         % like compute_dhdx_u except propagating output vector backwards
         % (pagefun over output units
      end
      
      
   end   
end

