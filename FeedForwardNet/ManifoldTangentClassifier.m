classdef ManifoldTangentClassifier < FeedForwardNet
   
   properties
      tangentCoeff % coefficient for tangent penalty term
   end
   
   methods
      function obj = ManifoldTangentClassifier(tangentCoeff, varargin)
         obj = obj@FeedForwardNet(varargin{:});
         obj.tangentCoeff = tangentCoeff;
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
         [y, z] = obj.feed_forward(x, mask);
         
         % get outputLayer output and backpropagate loss
         [grad, output, dLdx] = obj.backprop(x, y, t, z, mask);
         
         % compute mainfold tangent penalty gradient and add it to grad
         penalty = obj.compute_penalty_gradient(x, u, y, z, output, mask);
         penalty = obj.unroll_gradient(penalty);
         grad = cellfun(@plus, grad, penalty, 'UniformOutput', false);
      end
      
      function penalty = compute_penalty_gradient(obj, x, u, y, z, output, mask)
         % Need to modify for maxout networks
         nHiddenLayers = length(obj.hiddenLayers);
         
         % Compute Dy and D2y for each layer and also cumulative forward
         % Jacobian*u products
         Dy = cell(1, nHiddenLayers+1);
         D2y = cell(1, nHiddenLayers+1);
         dhdx_u = cell(1, nHiddenLayers);
         
         layer = obj.hiddenLayers{1};
         [Dy{1}, D2y{1}] = obj.compute_Dy_values(z{1}, y{1}, mask{2}, layer);
         dhdx_u{1} = obj.compute_dhdx_u(u, Dy{1}, layer);
         for i = 2:nHiddenLayers
            layer = obj.hiddenLayers{i};
            [Dy{i}, D2y{i}] = obj.compute_Dy_values(z{i}, y{i}, mask{i+1}, layer);
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
         penalty{1} = cell(1, 2);
         [penalty{1}{1}, penalty{1}{2}] = obj.compute_layer_penalty(x, dodx_u, dodh{1}, ...
                                                                     u, Dy{1}, D2y{1}, layer);
                                                                  
         % other hiddenLayers                                                         
         for i = 2:nHiddenLayers
            layer = obj.hiddenLayers{i};
            penalty{i} = cell(1, 2);
            [penalty{i}{1}, penalty{i}{2}] = obj.compute_hiddenLayer_penalty(y{i-1}, dodx_u, dodh{i}, ...
                                                                     dhdx_u{i-1}, Dy{i}, D2y{i}, layer);
         end
         
         % outputLayer
         layer = obj.outputLayer;
         penalty{end} = cell(1, 2);
         [penalty{end}{1}, penalty{end}{2}] = obj.compute_outputLayer_penalty(y{end}, output, dodx_u, 1, ...
                                                                     dhdx_u{nHiddenLayers}, Dy{end}, D2y{end}, layer);
         
      end
      
      function [Dy, D2y] = compute_Dy_values(obj, z, y, mask, layer)
         Dy = layer.compute_Dy(obj, z, y);
         if obj.isDropout
            if layer.isDiagonalDy
               Dy = Dy.*mask;
            else
               Dy = bsxfun(@times, Dy, shiftdim(mask', -1));
            end
         end 
         
         D2y = [];
         if ~layer.isLocallyLinear
            D2y = layer.compute_D2y(z, y, Dy);
            if obj.isDropout
               if layer.isDiagonalDy
                  D2y = D2y.*mask;
               else % D2y ~ j x k x n x i (L2 x L2 x N x L2) for dy^n_i / dz_j dz_k 
                  D2y = bsxfun(@times, D2y, shiftdim(mask', -2));
               end
            end
         end
      end
      
      function [W_pen, b_pen] = compute_hiddenLayer_penalty(obj, x, dodx_u, dodh, ...
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
         
         if layer.isLocallyLinear 
            sum1 = 0;
            b_pen = 0;
         else % Must include terms with D2y as a factor
            temp2 = bsxfun(@times, temp1, D2y); % L2 x N x U x C
            temp3 = bsxfun(@times, temp2, pagefun(@mtimes, layer.params{1}, dhdx_u)); % L2 x N x U x C
            prod1 = pagefun(@mtimes, temp3, x')/N; % L2 x L1 x U x C
            sum1 = sum(sum(prod1, 4), 3); % L2 x L1
            
            temp4 = mean(temp3, 2);
            b_pen = obj.tangentCoeff*sum(sum(temp4, 4), 3);
         end
         
         temp5 = bsxfun(@times, temp1, Dy); % L2 x N x U x C
         prod2 = pagefun(@mtimes, temp5, permute(dhdx_u, [2, 1, 3]))/N; % L2 x L1 x U x C
         sum2 = sum(sum(prod2, 4), 3);
         W_pen = obj.tangentCoeff*(sum1 + sum2);                                                                                                  
      end
      
      function [W_pen, b_pen] = compute_outputLayer_penalty(obj, x, dodx_u, dhdx_u, ...
                                                               Dy, D2y, layer)
         
         % (L2 = C) for outputLayer
         % x ~ L1 x N
         % y ~ C x N
         % dodx_u ~ 1 x N x U x C
         % dhdx_u ~ L1 x N x U
         % Dy ~ C x N or C x N x C
         % D2y ~ C x N or C x C x N x C                                      
         N = size(x, 2);
         if layer.isDiagonalDy  % Just like compute_hiddenLayer_penalty with dodh = I
            if layer.isLocallyLinear
               sum1 = 0;
               b_pen = 0;
            else % Need to include second derivative terms
               temp1 = bsxfun(@times, dodx_u, D2y); % C x N x U x C
               temp2 = bsxfun(@times, temp1, pagefun(@mtimes, layer.params{1}, dhdx_u));
               prod1 = pagefun(@mtimes, temp2, x')/N;
               sum1 = sum(sum(prod1, 4), 3);
               
               temp3 = mean(temp2, 2);
               b_pen = obj.tangentCoeff*sum(sum(temp3, 4), 3);
            end
            
            temp4 = bsxfun(@times, dodx_u, Dy);
            prod2 = pagefun(@mtimes, temp4, permute(dhdx_u, [2, 1, 3]))/N;
            sum2 = sum(sum(prod2, 4), 3);
            W_pen = obj.tangentCoeff*(sum1 + sum2);
         else % Jacobian is full matrix, D2y is 3D Hessian like in softmax (assume ~isLocallyLinear)
            temp1 = pagefun(@mtimes, layer.params{1}, dhdx_u); % C x N x U
            temp2 = squeeze(pagefun(@mtimes, permute(D2y, [1 2 3 5 4]), ...
                                    permute(temp1, [1 4 2 3 5]))); % C x N x U x C from (C x C x N x 1 x C mtimes C x 1 x N x U x 1)
            temp3 = bsxfun(@times, temp2, dodx_u); % C x N x U x C
            prod1 = pagefun(@mtimes, temp3, x')/N; % C x L1 x U x C
            sum1 = sum(sum(prod1, 4), 3);
            
            temp4 = mean(temp3, 2);
            b_pen = obj.tangentCoeff*sum(sum(temp4, 4), 3);
            
            temp5 = bsxfun(@times, dodx_u, permute(Dy, [1 2 4 3])); % C x N x U x C
            prod2 = pagefun(@mtimes, temp5, permute(dhdx_u, [2 1 3])); % C x L1 x U x C
            sum2 = sum(sum(prod2, 4), 3);
            W_pen = obj.tangentCoeff*(sum1 + sum2);
         end
      end
      
      function dhdx_u = compute_dhdx_u(~, prev_dhdx_u, Dy, layer)
         % prev_dhdx_u ~ L1 x N x U
         
         if ismatrix(layer.params{1})
            W_times_prev = pagefun(@mtimes, layer.params{1}, prev_dhdx_u);
         else % maxout layer
            W_times_prev = pagefun(@mtimes, permute(layer.params{1}, [1, 2, 4, 3]), ...
                                       prev_dhdx_u); % L2 x N x U x k
         end
         
         if layer.isDiagonalDy
            if ismatrix(layer.params{1})
               dhdx_u = bsxfun(@times, Dy, W_times_prev);
            else % maxout layer
               dhdx_u = sum(bsxfun(@times, permute(Dy, [1 2 4 3]), W_times_prev), 4);
            end
         else % Dy ~ z x N x y (softmax layer or similar)
            dhdx_u = squeeze(pagefun(@mtimes, permute(Dy, [3, 1, 2]), ...
                           permute(W_times_prev, [1 4 2 3])));  % probably can be optimized better
         end
      end
      
      function dodh = compute_dodh(~, prev_dodh, Dy, layer)
         % like compute_dhdx_u except propagating output vector backwards
         % (pagefun over output units)
         
         % prev_dodh ~ L2 x N x C
         
         if layer.isDiagonalDy
            if ismatrix(layer.params{1})
               dodz = bsxfun(@times, Dy, prev_dodh); % L2 x N x C
            else % maxout layer
               dodz = bsxfun(@times, permute(Dy, [1 2 4 3]), prev_dodh); % L2 x N x C x k
            end
         else % Dy is full matrix ~ z x N x y (softmax layer or similar)
            dodz = squeeze(pagefun(@mtimes, permute(Dy, [1 3 2]), permute(prev_dodh, [1 4 2 3]))); % L2 x N x C
         end
         
         if ismatrix(layer.params{1})
            dodh = pagefun(@mtimes, layer.params{1}', dodz); % L1 x N x C
         else % maxout layer
            dodh = sum(pagefun(@mtimes, permute(layer.params{1}, [2 1 4 3]), dodz), 4);
         end
      end

   end   
end

