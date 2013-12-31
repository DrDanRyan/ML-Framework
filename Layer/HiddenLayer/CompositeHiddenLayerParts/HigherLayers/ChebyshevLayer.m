classdef ChebyshevLayer < ParamsFunctions & matlab.mixin.Copyable
   
   properties
      % params = f ~ L x 1 x D x R
      L % number of hidden unit groups
      D % dimension of hidden unit groups
      R % resolution of Chebyshev polynomial
      
      DfMatrix % ~ R x R derivative transform matrix
      xCheb % Chebyshev points ~ 1 x 1 x 1 x R
      wCheb % Lagrange interpolation weights ~ 1 x 1 x 1 x R
      
      % Temp values stored during forward pass used for backprop
      dydx
      dydf 
   end
   
   methods
      function obj = ChebyshevLayer(L, D, R, varargin)
         obj = obj@ParamsFunctions(varargin{:});         
         obj.L = L;
         obj.D = D;
         obj.R = R;
         [x, w, obj.DfMatrix] = compute_cheb_constants(R, obj.gpuState); 
         obj.xCheb = shiftdim(x, -3);
         obj.wCheb = shiftdim(w, -3); 
         obj.init_params();   
      end
      
      function init_params(obj)
         if isempty(obj.initScale)
            obj.initScale = .01;
         end
         obj.params{1} = obj.initScale*obj.gpuState.randn(obj.L, 1, obj.D, obj.R);
      end
      
      function y = feed_forward(obj, x, isSave)
         % assumes x is in the interval [-1, 1]
         w_over_x_minus_xCheb = bsxfun(@rdivide, obj.wCheb, ...
                                             bsxfun(@minus, x, obj.xCheb)); % L x N x D x R
         isReplaceVals = any(isinf(w_over_x_minus_xCheb(:)));
         denominator = sum(w_over_x_minus_xCheb, 4); % L x N x D
         y = bsxfun(@rdivide, sum(bsxfun(@times, obj.params{1}, w_over_x_minus_xCheb), 4), ...
                              denominator);
         if isReplaceVals
            mask = isinf(w_over_x_minus_xCheb);
            replacementVals = sum(bsxfun(@times, obj.params{1}, mask), 4);
            replacementIdx = isnan(y);
            y(replacementIdx) = replacementVals(replacementIdx);
            clear replacementVals
         end

         if nargin == 3 && isSave
            Df = permute(pagefun(@mtimes, obj.DfMatrix, ...
                              permute(obj.params{1}, [4, 1, 2, 3])), [2, 3, 4, 1]); % L x 1 x D x R
            obj.dydx = bsxfun(@rdivide, sum(bsxfun(@times, Df, w_over_x_minus_xCheb), 4), ...
                                        denominator); % L x N x D
            if isReplaceVals
               replacementVals = sum(bsxfun(@times, Df, mask), 4);
               obj.dydx(replacementIdx) = replacementVals(replacementIdx);
               clear replacementVals replacementIdx
            end
            clear Df
            
            obj.dydf = bsxfun(@rdivide, w_over_x_minus_xCheb, denominator); % L x N x D x R
            clear denominator w_over_x_minus_xCheb
            if isReplaceVals
               replacementIdx = isnan(obj.dydf);
               obj.dydf(replacementIdx) = mask(replacementIdx);
            end
         end
      end
      
      function [grad, dLdx] = backprop(obj, dLdy)
         dLdx = obj.dydx.*dLdy;
         obj.dydx = [];
         grad = {mean(bsxfun(@times, dLdy, obj.dydf), 2)};
         obj.dydf = [];
      end
      
      function gather(obj)
         obj.params{1} = gather(obj.params{1});
         obj.DfMatrix = gather(obj.DfMatrix);
         obj.xCheb = gather(obj.xCheb);
         obj.wCheb = gather(obj.wCheb);
         obj.gpuState.isGPU = false;
      end
      
      function push_to_GPU(obj)
         obj.params{1} = single(gpuArray(obj.params{1}));
         obj.DfMatrix = single(gpuArray(obj.DfMatrix));
         obj.xCheb = single(gpuArray(obj.xCheb));
         obj.wCheb = single(gpuArray(obj.wCheb));
         obj.gpuState.isGPU = true;
      end
      
   end   
end

