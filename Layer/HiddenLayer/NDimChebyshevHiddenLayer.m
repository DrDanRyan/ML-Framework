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
      
      function [grad, dLdx, y] = backprop(obj, x, y, z, dLdy)
         N = size(x, 2);
         [Dy, dydf, dydsigma] = obj.compute_Dy(z, y);
         dLdz = bsxfun(@times, dLdy, Dy); % L2 x N x 1 x cDim
         dLdx = sum(pagefun(@mtimes, permute(obj.params{1}, [2 1 3 4]), dLdz), 4);
         grad{1} = pagefun(@mtimes, dLdz, x')/N;
         grad{2} = mean(dLdz, 2);
         grad{3} = mean(bsxfun(@times, dLdy, dydf), 2);
         grad{4} = mean(bsxfun(@times, dLdy, dydsigma), 2);
      end
      
      function [y, z] = feed_forward(obj, x)
         z = obj.compute_z(x);
         v = exp(-2*z);
         u = 2./(1 + v);
         zHat = u - 1; % robust version of tanh(z)
         y = obj.compute_Chebyshev_interpolants(zHat);
      end
      
      function y = compute_Chebyshev_interpolants(obj, zHat)
         zHat_minus_xCheb = bsxfun(@minus, zHat, obj.xCheb); % L2 x N x 1 x cDim x cRes
         isReplaceVals = any(any(any(any(zHat_minus_xCheb == 0))));
         
         denom_sum = sum(bsxfun(@rdivide, obj.wCheb, zHat_minus_xCheb), 5); % L2 x N x 1 x cDim
         if isReplaceVals
            denom_sum(denom_sum == Inf | denom_sum == -Inf) = 1; % correct for cases where z == xCheb
         end
         denom_product = prod(denom_sum, 4); % L2 x N
        
         num_sum = sum(bsxfun(@rdivide, bsxfun(@times, obj.wCheb, obj.params{3}), ...
                                 zHat_minus_xCheb), 5); % L2 x N x cRank x cDim
         if isReplaceVals
            mask = obj.gpuState.make_numeric(zHat_minus_xCheb == 0); % L2 x N x 1 x cDim x cRes
            num_replace = sum(bsxfun(@times, obj.params{3}, mask), 5); % L2 x N x cRank x cDim
            replaceIdx = num_sum == Inf | num_sum == -Inf;
            num_sum(replaceIdx) = num_replace(replaceIdx);
         end
         num_product = prod(num_sum, 4); % L2 x N x cRank
         num_prod_lincombo = sum(bsxfun(@times, obj.params{4}, num_product), 3); % L2 x N
         
         y = num_prod_lincombo./denom_product;
      end
      
      function value = compute_z(obj, x)
         % z ~ L2 x N x 1 x cDim
         value = bsxfun(@plus, pagefun(@mtimes, obj.params{1}, x), obj.params{2});
      end
      
      function [Dy, dydf, dydsigma] = compute_Dy(obj, z, y)
         % Dy ~ L2 x N x 1 x cDim
         % dydf ~ L2 x N x cRank x cDim x cRes
         % dydsigma ~ L2 x N x cRank
         v = exp(-2*z);
         u = 2./(1 + v);
         zHat = u - 1; % robust version of tanh(z)
         dydzHat = v.*u.*u; % robust Dy for tanh layer
         
         zHat_minus_xCheb = bsxfun(@minus, zHat, obj.xCheb); % L2 x N x 1 x cDim x cRes
         isReplaceVals = any(any(any(any(zHat_minus_xCheb == 0))));
         if isReplaceVals
            replaceIdx = repmat(obj.gpuState.make_numeric(any(zHat_minus_xCheb == 0, 5)), ...
                                    1, 1, obj.cRank, 1); % L2 x N x cRank x cDim
         end
         
         Df = pagefun(@mtimes, obj.D1, permute(obj.params{3}, [5 1 2 3 4]));
         Df = permute(Df, [2 3 4 5 1]); % L2 x 1 x cRank x cDim x cRes
         
         wCheb_over_zHat_minus_xCheb = bsxfun(@rdivide, obj.wCheb, zHat_minus_xCheb); % L2 x N x 1 x cDim x cRes
         denom_sum = sum(wCheb_over_zHat_minus_xCheb, 5); % L2 x N x 1 x cDim
         if isReplaceVals
            denom_sum(zHat_minus_xCheb==0) = 1; % correct for cases where z == xCheb (will also replace value in numerator)
         end
         denom_product = prod(denom_sum, 4); % L2 x N
         
         Dy = obj.gpuState.zeros([size(y), 1, obj.cDim]);
         dydf = obj.gpuState.zeros([size(y), obj.cRank, obj.cDim, obj.cRes]);
         num_sum = sum(bsxfun(@rdivide, bsxfun(@times, obj.wCheb, obj.params{3}), ...
                                 zHat_minus_xCheb), 5); % L2 x N x cRank x cDim
         num_product = prod(num_sum, 4);
         dydsigma = bsxfun(@rdivide, num_product, denom_product); % L2 x N x cRank
         
         for d = 1:obj.cDim
            % Dy
            temp_num_sum = num_sum;
            temp_num_sum(:,:,:,d) = sum(bsxfun(@rdivide, bsxfun(@times, obj.wCheb, Df(:,:,:,d,:)), ...
                                 zHat_minus_xCheb(:,:,:,d)), 5);
            if isReplaceVals
               mask = obj.gpuState.make_numeric(zHat_minus_xCheb == 0); % L2 x N x 1 x cDim x cRes
               num_replace = sum(bsxfun(@times, obj.params{3}, mask), 5); % L2 x N x cRank x cDim
               num_replace(:,:,:,d) = sum(bsxfun(@times, Df(:,:,:,d,:), mask), 5);
               temp_num_sum(replaceIdx) = num_replace(replaceIdx);
            end
            num_product = prod(temp_num_sum, 4); % L2 x N x cRank
            num_prod_lincombo = sum(bsxfun(@times, obj.params{4}, num_product), 3); % L2 x N
            Dy(:,:,1,d) = num_prod_lincombo./denom_product; % L2 x N
            
            % dydf
            temp_num_sum = num_sum;
            temp_num_sum(:,:,:,d) = 1;
            num_product = prod(temp_num_sum, 4); % L2 x N x cRank
            num_part2 = wCheb_over_zHat_minus_xCheb(:,:,1,d,:); % L2 x N x 1 x 1 x cRes
            if isReplaceVals % TODO fix this line, need to make rest of cRes vals = 0 when replacing Inf value with 1
               num_part2(num_part2 == Inf | num_part2 == -Inf) = 1;
            end
            numerator = bsxfun(@times, obj.params{4}, bsxfun(@times, num_product, num_part2));
            dydf(:,:,:,d,:) = bsxfun(@rdivide, numerator, denom_product); % L2 x N x cRank x 1 x cRes
         end
         Dy = Dy.*dydzHat;
      end
      
      function value = compute_D2y(obj, z, y, Dy)
         % pass
      end
         
      function init_params(obj)
         r = obj.initScale;
         obj.params{1} = 2*r*obj.gpuState.rand([obj.outputSize, obj.inputSize, 1, obj.cDim]) - r;
         obj.params{2} = obj.gpuState.zeros([obj.outputSize, 1, 1, obj.cDim]);
         obj.params{3} = 2*obj.gpuState.rand([obj.outputSize, 1, obj.cRank, obj.cDim, obj.cRes]) - 1;
         obj.params{4} = obj.gpuState.ones([obj.outputSize, 1, obj.cRank]);
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

