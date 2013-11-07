classdef SoftmaxOutputLayer < StandardOutputLayer
   % TODO: improve robustness using "Tricks of the Trade" 2nd ed Ch 11
   % tricks
   properties 
      isLocallyLinear = false
      isDiagonalDy = false
   end
   
   methods
      function obj = SoftmaxOutputLayer(inputSize, outputSize, varargin)
         obj = obj@StandardOutputLayer(inputSize, outputSize, varargin{:});
      end
      
      function [dLdz, y] = compute_dLdz(obj, x, t)
         y = obj.feed_forward(x);
         dLdz = y - t;
      end
      
      function value = compute_Dy(~, ~, y)
         % in non-batch mode would return: value = diag(y) - y*y' ~ C x C
         C = size(y, 1);
         id13 = reshape(eye(C), [C, 1, C]);
         yT = shiftdim(y', -1);
         value = bsxfun(@times, y, id13) - bsxfun(@times, y, yT); % C x N x C
      end
      
      function value = compute_D2y(~, ~, y, Dy)
         % In non-batch mode would return: d2y_k/(dz_i dz_j) ~ C x C x C
         % For batch mode will return with shape C x C x N x C (note it is
         % completely symmetric in all C dimensions)
         [C, N] = size(y);
         diagTerm = bsxfun(@times, shiftdim(Dy, -1), eye(C));
         Dy_yT_term = bsxfun(@times, shiftdim(Dy, -1), reshape(y, [C, 1, N, 1]));
         y_Dy_term = bsxfun(@times, reshape(Dy, [C, 1, N, C]), reshape(y, [1, C, N, 1]));
         value = diagTerm - Dy_yT_term - y_Dy_term; % C x C x N x C 
      end
      
      function y = feed_forward(obj, x)
         z = obj.compute_z(x);
         y = softmax(z);
      end
   
      function loss = compute_loss(~, y, t)
         loss = mean(sum(-t.*log(y), 1));
      end 
   end
   
end

