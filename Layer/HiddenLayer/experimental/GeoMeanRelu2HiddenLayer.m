classdef GeoMeanRelu2HiddenLayer < StandardLayer & HiddenLayer
   
   properties
      D
      % parmas = {W, b} with W ~ L2 x L1 x D and b ~ L2 x 1 x D
      isLocallyLinear = false
   end
   
   methods
      function obj = GeoMeanRelu2HiddenLayer(inputSize, outputSize, D, varargin)
         obj = obj@StandardLayer(inputSize, outputSize, varargin{:});
         obj.D = D;
         obj.init_params();
      end
      
      function init_params(obj)
         obj.params{2} = .01*obj.gpuState.ones(obj.outputSize, 1, obj.D);
         for d = 1:obj.D
            obj.params{1}(:,:,d) = matrix_init(obj.outputSize, obj.inputSize, obj.initType, ...
                                                      obj.initScale, obj.gpuState);
         end
      end
      
      function [y, zRect] = feed_forward(obj, x)
         z = obj.compute_z(x);
         zRect = max(0, z);
         y = prod(zRect, 3).^(1/obj.D);
      end
      
      function value = compute_Dy(obj, zRect, y)
         value = bsxfun(@rdivide, y, obj.D*zRect);
         value(isnan(value)) = 0;
      end
      
      function value = compute_D2y(obj, ffExtras, y, Dy)
         % pass
      end
      
      function value = compute_z(obj, x)
         value = bsxfun(@plus, pagefun(@mtimes, obj.params{1}, x), obj.params{2});
      end
      
      function [grad, dLdx, Dy] = backprop(obj, x, y, zRect, dLdy)
         Dy = obj.compute_Dy(zRect, y);
         dLdz = bsxfun(@times, dLdy, Dy);
         dLdx = sum(pagefun(@mtimes, permute(obj.params{1}, [2 1 3]), dLdz), 3);
         grad{1} = pagefun(@mtimes, dLdz, x')/size(x, 2); % L2 x L1 x D
         grad{2} = mean(dLdz, 2); % L2 x 1 x D
      end
   end
   
end

