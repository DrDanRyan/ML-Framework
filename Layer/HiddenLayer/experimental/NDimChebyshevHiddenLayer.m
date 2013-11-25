classdef NDimChebyshevHiddenLayer < HiddenLayer
    
   properties
      % params = {W, b, f, sigma} where:
      % W ~ L2 x L1 x 1 x cDim    b ~ L2 x 1 x 1 x cDim
      % f ~ L2 x 1 x cRank x cDim x cRes   sigma ~ L2 x 1 x cRank
      
      params
      inputSize
      outputSize
      cDim
      cRes
      cRank
      
      D1  % ~ cRes x cRes
      xCheb % ~ 1 x 1 x 1 x 1 x cRes
      wCheb % ~ 1 x 1 x 1 x 1 x cRes
      
      initScale
      gpuState
      isLocallyLinear = false;
   end
    
   methods
      function obj = NDimChebyshevHiddenLayer(inputSize, outputSize, cRank, cDim, cRes, varargin)
         p = inputParser();
         p.addParamValue('initScale', .005);
         p.addParamValue('gpu', []);
         parse(p, varargin{:});
         
         obj.inputSize = inputSize;
         obj.outputSize = outputSize;
         obj.cDim = cDim;
         obj.cRes = cRes;
         obj.cRank = cRank;
         obj.gpuState = GPUState(p.Results.gpu);
         obj.initScale = p.Results.initScale;
         [x, w, obj.D1] = compute_cheb_constants(cRes, obj.gpuState); 
         obj.xCheb = permute(x, [2 3 4 5 1]);
         obj.wCheb = permute(w, [2 3 4 5 1]); 
         obj.init_params();   
      end
      
      function [grad, dLdx, y] = backprop(obj, x, y, ffExtras, dLdy)
         N = size(x, 2);
         [Dy, dydf, dydsigma] = obj.compute_Dy(ffExtras, y);
         dLdz = bsxfun(@times, dLdy, Dy); % L2 x N x 1 x cDim
         dLdx = sum(pagefun(@mtimes, permute(obj.params{1}, [2 1 3 4]), dLdz), 4);
         grad{1} = pagefun(@mtimes, dLdz, x')/N;
         grad{2} = mean(dLdz, 2);
         grad{3} = mean(bsxfun(@times, dLdy, dydf), 2);
         grad{4} = mean(bsxfun(@times, dLdy, dydsigma), 2);
      end
      
      function [y, ffExtras] = feed_forward(obj, x)
         z = obj.compute_z(x);
         v = exp(-2*z);
         u = 2./(1 + v);
         tanhz = u - 1; % robust version of tanh(z)
         dtanhz_dz = v.*u.*u;
         [y, chebRank1, cheb1D] = obj.compute_Chebyshev_interpolants(tanhz);
         ffExtras = {tanhz, dtanhz_dz, cheb1D, chebRank1};    
      end
      
      function value = compute_z(obj, x)
         % z ~ L2 x N x 1 x cDim
         value = bsxfun(@plus, pagefun(@mtimes, obj.params{1}, x), obj.params{2});
      end
      
      function [y, chebRank1, cheb1D] = compute_Chebyshev_interpolants(obj, tanhz)
         tanhz_minus_xCheb = bsxfun(@minus, tanhz, obj.xCheb); 
         isReplaceVals = any(any(any(any(tanhz_minus_xCheb == 0))));
         
         cheb1D_numerator = sum(bsxfun(@rdivide,  bsxfun(@times, obj.wCheb, obj.params{3}), ...
                                 tanhz_minus_xCheb), 5); % L2 x N x cRank x cDim
         cheb1D_denominator = sum(bsxfun(@rdivide, obj.wCheb, tanhz_minus_xCheb), 5); % L2 x N x 1 x cDim
         cheb1D = bsxfun(@rdivide, cheb1D_numerator, cheb1D_denominator); % L2 x N x cRank x cDim
         clear cheb1D_numerator cheb1D_denominator
         if isReplaceVals
            mask = obj.gpuState.make_numeric(tanhz_minus_xCheb == 0); % L2 x N x 1 x cDim x cRes
            clear tanhz_minus_xCheb
            replacementVals = sum(bsxfun(@times, obj.params{3}, mask), 5); % L2 x N x cRank x cDim
            clear mask
            replaceIdx = isnan(cheb1D);
            cheb1D(replaceIdx) = replacementVals(replaceIdx);
            clear replaceIdx replacementVals
         end
         
         chebRank1 = prod(cheb1D, 4); % L2 x N x cRank
         y = sum(bsxfun(@times, chebRank1, obj.params{4}), 3); % L2 x N
      end
      
      function [Dy, dydf, dydsigma] = compute_Dy(obj, ffExtras, y)
         % Dy ~ L2 x N x 1 x cDim
         % dydf ~ L2 x N x cRank x cDim x cRes
         [tanhz, dtanhz_dz, cheb1D, chebRank1] = ffExtras{:};
         dydsigma = chebRank1;
         tanhz_minus_xCheb = bsxfun(@minus, tanhz, obj.xCheb); % L2 x N x 1 x cDim x cRes
         
         Dy = obj.gpuState.zeros([size(y), 1, obj.cDim]); % L2 x N x 1 x cDim
         dydf = obj.gpuState.zeros([size(y), obj.cRank, obj.cDim, obj.cRes]); % L2 x N x cRank x cDim x cRes
         for d = 1:obj.cDim
            temp = cheb1D;
            temp(:,:,:,d) = 1;
            partial_prod = prod(temp, 4); % L2 x N x cRank
            clear temp
            
            % compute Dy for this cDim           
            f_d = obj.params{3}(:,:,:,d,:); % L2 x 1 x cRank x 1 x cRes
            Df = permute(pagefun(@mtimes, obj.D1, permute(f_d, [5 1 2 3 4])), [2 3 4 5 1]); % L2 x 1 x cRank x 1 x cRes
            clear f_d
            Df_term_numer = sum(bsxfun(@rdivide, bsxfun(@times, Df, obj.wCheb), ...
                                 tanhz_minus_xCheb(:,:,:,d,:)), 5); % L2 x N x cRank
            Df_term_denom = sum(bsxfun(@rdivide, obj.wCheb, tanhz_minus_xCheb(:,:,:,d,:)), 5); % L2 x N
            Df_term = bsxfun(@rdivide, Df_term_numer, Df_term_denom); % L2 x N x cRank
            clear Df_term_numer Df_term_denom
            if any(any(any(isnan(Df_term))))
               mask = obj.gpuState.make_numeric(tanhz_minus_xCheb(:,:,:,d,:) == 0);
               replacementVals = sum(bsxfun(@times, Df, mask), 5);
               replacementIdx = isnan(Df_term);
               Df_term(replacementIdx) = replacementVals(replacementIdx);
               clear replacementVals replacementIdx
            end
            Dy_prod = partial_prod.*Df_term;
            Dy(:,:,1,d) = sum(bsxfun(@times, Dy_prod, obj.params{4}), 3); % still need to mult by dtanhz_dz
            clear Dy_prod
            
            % compute dydf for this cDim
            no_f_term_numer = bsxfun(@rdivide, obj.wCheb, tanhz_minus_xCheb(:,:,:,d,:));
            no_f_term_denom = sum(no_f_term_numer, 5);
            no_f_term = bsxfun(@rdivide, no_f_term_numer, no_f_term_denom); % L2 x N x 1 x 1 x cRes
            clear no_f_term_numer no_f_term_denom
            if any(any(any(isnan(no_f_term))))
               % keep same mask as computed in Dy case
               % replacementVals = mask
               replacementIdx = isnan(no_f_term);
               no_f_term(replacementIdx) = mask(replacementIdx);
               clear replacementIdx
            end
            dydf_prod = bsxfun(@times, no_f_term, partial_prod);
            dydf(:,:,:,d,:) = bsxfun(@times, dydf_prod, obj.params{4});
            clear dydf_prod
         end
         Dy = Dy.*dtanhz_dz; % L2 x N x 1 x cDim
      end
      
      function value = compute_D2y(obj, z, y, Dy)
         % pass
      end
         
      function init_params(obj)
         r = obj.initScale;
         obj.params{1} = 2*r*obj.gpuState.rand([obj.outputSize, obj.inputSize, 1, obj.cDim]) - r;
         obj.params{2} = obj.gpuState.zeros([obj.outputSize, 1, 1, obj.cDim]);
         obj.params{3} = .1*obj.gpuState.randn([obj.outputSize, 1, obj.cRank, obj.cDim, obj.cRes]);
         obj.params{4} = obj.gpuState.ones([obj.outputSize, 1, obj.cRank])/obj.cRank;
      end
      
      function gather(obj)
         obj.params = cellfun(@(p) gather(p), obj.params, 'UniformOutput', false);
         obj.gpuState.isGPU = false;
      end
      
      function push_to_GPU(obj)
         obj.params = cellfun(@(p) single(gpuArray(p)), obj.params, 'UniformOutput', false);
         obj.gpuState.isGPU = true;
      end
      
      function increment_params(obj, delta)
         obj.params = cellfun(@plus, obj.params, delta, 'UniformOutput', false);
      end
   end
    
end

