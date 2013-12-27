classdef ChebyshevRank1HiddenLayer < HiddenLayer & matlab.mixin.Copyable & ParamsFunctions
    
   properties
      % params = {W, b, f} where:
      % W ~ L2 x L1 x 1 x D    b ~ L2 x 1 x D
      % f ~ L2 x 1 x D x R
      inputSize
      outputSize
      D  % dimension of Chebyshev approx
      R  % resolution of Chebyshev approx
      
      D1  % ~ R x R
      xCheb % ~ 1 x 1 x 1 x R
      wCheb % ~ 1 x 1 x 1 x R
      
      % Temp values stored during forward pass used for backprop
      dydz
      dydf
   end
    
   methods
      function obj = ChebyshevRank1HiddenLayer(inputSize, outputSize, D, R, varargin)
         obj = obj@ParamsFunctions(varargin{:});         
         obj.inputSize = inputSize;
         obj.outputSize = outputSize;
         obj.D = D;
         obj.R = R;
         [x, w, obj.D1] = compute_cheb_constants(R, obj.gpuState); 
         obj.xCheb = shiftdim(x, -3);
         obj.wCheb = shiftdim(w, -3); 
         obj.init_params();   
      end
      
      function init_params(obj)
         obj.params{1} = obj.gpuState.nan(obj.outputSize, obj.inputSize, obj.D);
         obj.params{2} = obj.gpuState.zeros(obj.outputSize, 1, obj.D);
         for d = 1:obj.D
            obj.params{1}(:,:,d) = matrix_init(obj.outputSize, obj.inputSize, obj.initType, ...
                                                   obj.initScale, obj.gpuState);
         end
         
         obj.params{3} = .01*obj.gpuState.randn(obj.outputSize, 1, obj.D, obj.R);
      end
      
      function y = feed_forward(obj, x, isSave)
         v = exp(-2*obj.compute_z(x));
         u = 2./(1 + v);
         tanhz = u - 1;
         dtanhz_dz = v.*u.*u;
         clear u v
         tanhz_minus_xCheb = bsxfun(@minus, tanhz, obj.xCheb); 
         isReplaceVals = any(tanhz_minus_xCheb(:) == 0);
         
         cheb1D_numerator = sum(bsxfun(@rdivide,  bsxfun(@times, obj.wCheb, obj.params{3}), ...
                                 tanhz_minus_xCheb), 4); % L2 x N x D
         cheb1D_denominator = sum(bsxfun(@rdivide, obj.wCheb, tanhz_minus_xCheb), 4); % L2 x N x D
         cheb1D = bsxfun(@rdivide, cheb1D_numerator, cheb1D_denominator); % L2 x N x D
         clear cheb1D_numerator cheb1D_denominator
         if isReplaceVals
            mask = tanhz_minus_xCheb == 0; % L2 x N x D x R
            replacementVals = sum(bsxfun(@times, obj.params{3}, mask), 4); % L2 x N x D
            replaceIdx = isnan(cheb1D);
            cheb1D(replaceIdx) = replacementVals(replaceIdx);
            clear replaceIdx replacementVals
         end
         y = prod(cheb1D, 3); % L2 x N
         
         if nargin == 3 && isSave
            dy_dtanhz = obj.gpuState.zeros([size(y), obj.D]); % L2 x N x D
            obj.dydf = obj.gpuState.zeros([size(y), obj.D, obj.R]); % L2 x N x D x R
            for d = 1:obj.D
               temp = cheb1D;
               temp(:,:,d) = 1;
               partial_prod = prod(temp, 3); % L2 x N
               clear temp

               % compute Dy for this dimension           
               f_d = permute(obj.params{3}(:,:,d,:), [4, 1, 2, 3]); % R x L2 x 1 x 1
               Df = permute(pagefun(@mtimes, obj.D1, f_d), [2 3 4 1]); % L2 x 1 x 1 x R
               clear f_d
               Df_term_numer = sum(bsxfun(@rdivide, bsxfun(@times, Df, obj.wCheb), ...
                                    tanhz_minus_xCheb(:,:,d,:)), 4); % L2 x N
               Df_term_denom = sum(bsxfun(@rdivide, obj.wCheb, tanhz_minus_xCheb(:,:,d,:)), 4); % L2 x N
               Df_term = bsxfun(@rdivide, Df_term_numer, Df_term_denom); % L2 x N
               clear Df_term_numer Df_term_denom
               if any(isnan(Df_term(:)))
                  replacementVals = sum(bsxfun(@times, Df, mask), 4);
                  replacementIdx = isnan(Df_term);
                  Df_term(replacementIdx) = replacementVals(replacementIdx);
                  clear replacementVals replacementIdx
               end
               dy_dtanhz(:,:,d) = partial_prod.*Df_term;

               % compute dydf for this dimension
               no_f_term_numer = bsxfun(@rdivide, obj.wCheb, tanhz_minus_xCheb(:,:,d,:));
               no_f_term_denom = sum(no_f_term_numer, 4);
               no_f_term = bsxfun(@rdivide, no_f_term_numer, no_f_term_denom); % L2 x N x 1 x R
               clear no_f_term_numer no_f_term_denom
               if any(isnan(no_f_term(:)))
                  replacementIdx = isnan(no_f_term);
                  no_f_term(replacementIdx) = mask(replacementIdx);
                  clear replacementIdx
               end
               obj.dydf(:,:,d,:) = bsxfun(@times, no_f_term, partial_prod);
            end
            obj.dydz = dy_dtanhz.*dtanhz_dz; % L2 x N x D
         end
      end
      
      function [grad, dLdx] = backprop(obj, x, ~, dLdy)
         grad{3} = mean(bsxfun(@times, dLdy, obj.dydf), 2);
         obj.dydf = [];
         
         dLdz = bsxfun(@times, dLdy, obj.dydz); % L2 x N x D
         obj.dydz = [];
         dLdx = sum(pagefun(@mtimes, permute(obj.params{1}, [2 1 3]), dLdz), 3);
         grad{1} = pagefun(@mtimes, dLdz, x')/size(x, 2);
         grad{2} = mean(dLdz, 2);
      end

      function value = compute_z(obj, x)
         % z ~ L2 x N x D
         value = bsxfun(@plus, pagefun(@mtimes, obj.params{1}, x), obj.params{2});
      end
   end
end

