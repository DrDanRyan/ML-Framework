classdef ChebyshevHiddenLayer < StandardHiddenLayer
   properties
      % params = {W, b, f} where f is warping function values at xCheb.  f ~ L2 x 1 x nCheb
      D1
      wCheb
      xCheb
      isLocallyLinear = false;
   end
   
   methods
      function obj = ChebyshevHiddenLayer(inputSize, outputSize, nChebPts, varargin)
         obj = obj@StandardHiddenLayer(inputSize, outputSize, varargin{:});
         [x, w, obj.D1] = compute_cheb_constants(nChebPts, obj.gpuState);
         obj.xCheb = permute(x, [2, 3, 1]); % 1 x 1 x nCheb
         obj.wCheb = permute(w, [2, 3, 1]); % 1 x 1 x nCheb
         
         % Initialize Chebyshev layer to be identity function
         obj.params{3} = permute(repmat(x, outputSize), [2, 3, 1]); % L2 x 1 x nCheb
      end
      
      function [grad, dLdx, Dy] = backprop(obj, x, y, z, dLdy)
         [Dy, dydf] = obj.compute_Dy(z, y);
         dLdz = dLdy.*Dy;
         dLdx = obj.params{1}'*dLdz;
         grad = obj.grad_from_dLdz(x, dLdz); % grad for W and b
         grad = [grad, mean(bsxfun(@times, dLdy, dydf), 2)]; % append grad for f
      end
      
      function [y, z] = feed_forward(obj, x)
         z = obj.compute_z(x);
         yHat = tanh(z);
         y = obj.compute_chebyshev_interpolations(yHat);
      end
      
      function y = compute_chebyshev_interpolations(obj, yHat)
         yHat_minus_xCheb = bsxfun(@minus, yHat, obj.xCheb);
         numerator = sum(bsxfun(@rdivide, bsxfun(@times, obj.wCheb, obj.params{3}), ...
                                       yHat_minus_xCheb), 3);
         denominator = sum(bsxfun(@rdivide, obj.wCheb, yHat_minus_xCheb), 3);
         y = numerator./denominator;
      end
      
      function [Dy, dydf] = compute_Dy(obj, z, ~)
         yHat = tanh(z);
         Df = obj.D1*permute(obj.params{3}, [3, 1, 2]); % nCheb x L2
         Df = permute(Df, [2 3 1]); % L2 x 1 x nCheb
         yHat_minus_xCheb = bsxfun(@minus, yHat, obj.xCheb);
         numerator = sum(bsxfun(@rdivide, bsxfun(@times, obj.wCheb, Df), ...
                                       yHat_minus_xCheb), 3);
         w_over_yHat_minus_xCheb = bsxfun(@rdivide, obj.wCheb, yHat_minus_xCheb);
         denominator = sum(w_over_yHat_minus_xCheb, 3);
         Dy = (numerator./denominator).*(1 - yHat).*(1 + yHat);
         dydf = bsxfun(@rdivide, w_over_yHat_minus_xCheb, denominator);
      end
   end
   
end

