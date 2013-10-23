classdef DAE < AutoEncoder
   % Denoising AutoEncoder
   
   properties     
      noiseType % string indicating type of input noise to use: 'none', 'dropout', 'Gaussian'
      noiseLevel % a scalar indicating the level of noise (i.e. dropout prob, or std_dev)
   end
   
   methods
      function obj = DAE(varargin)
         obj = obj@AutoEncoder(varargin{:});
         p = inputParser;
         p.addParamValue('noiseType', 'none');
         p.addParamValue('noiseLevel', .2);
         parse(p, varargin{:});
         
         obj.noiseType = p.Results.noiseType;
         obj.noiseLevel = p.Results.noiseLevel;
      end
      
      function [grad, xRecon] = gradient(obj, x, ~, ~)
         xCorrupt = obj.inject_noise(x);
         xCorrupt(isnan(x)) = 0;
         xCode = obj.encodeLayer.feed_forward(xCorrupt);
         if obj.isDropout
            mask = obj.gpuState.binary_mask(size(xCode), obj.dropout);
            xCode = xCode.*mask;
         else
            mask = [];
         end
         [decodeGrad, dLdxCode, xRecon] = obj.decodeLayer.backprop(xCode, x);
         if obj.isDropout
            dLdxCode = dLdxCode.*mask;
         end
         [encodeGrad, ~, dydz] = obj.encodeLayer.backprop(xCorrupt, xCode, dLdxCode);
               
         if ~isempty(obj.JacCoeff)
            encodeGrad{1} = encodeGrad{1} + ...
                              obj.compute_contraction_penalty(xCorrupt, dydz, mask);
         end
         
         if obj.isTiedWeights
            if ndims(encodeGrad{1}) <= 2
               grad = {encodeGrad{1}+decodeGrad{1}', encodeGrad{2}, decodeGrad{2}};
            else
               grad = {encodeGrad{1}+permute(decodeGrad{1}, [2, 1, 3]), ...
                        encodeGrad{2}, decodeGrad{2}};
            end
         else
            grad = [encodeGrad, decodeGrad];
         end
      end
      
      function penalty = compute_contraction_penalty(obj, x, dydz, mask)
         dydzSqMean = mean(dydz.*dydz, 2); % L2 x 1 (x k)
         penalty = obj.JacCoeff*bsxfun(@times, obj.encodeLayer.params{1}, dydzSqMean);
         
         if ~isempty(HessCoeff) % Should not be used in conjunction with dropout at this point
            [L1, N] = size(x);
            permvec = randperm(N, obj.HessBatchSize);
            dydzSample = dydz(:,permvec);
            xPerturbed = x(:,premvec) + obj.HessNoise*obj.gpuState.randn([L1, obj.HessBatchSize]);
            
         end    
      end
      
      function x = inject_noise(obj, x)
         switch obj.noiseType
            case 'none'
               % do nothing
            case 'dropout'
               x = x.*obj.gpuState.binary_mask(size(x), obj.noiseLevel);
            case 'Gaussian'
               x = x + obj.noiseLevel*obj.gpuState.randn(size(x));
         end
      end
      
      function loss = compute_loss(obj, xRecon, x)
         loss = obj.decodeLayer.compute_loss(xRecon, x);
      end
      
      function xCode = encode(obj, x)
         x(isnan(x)) = 0;
         xCode = obj.encodeLayer.feed_forward(x);
      end
      
      function xRecon = output(obj, x)
         x(isnan(x)) = 0;
         xCode = obj.encodeLayer.feed_forward(x);
         if obj.isDropout
            xCode = (1-obj.dropout)*xCode;
         end
         xRecon = obj.decodeLayer.feed_forward(xCode);
      end
      
      function increment_params(obj, delta_params)
         if obj.isTiedWeights
            obj.encodeLayer.increment_params(delta_params(1:2));
            obj.decodeLayer.increment_params({0, delta_params{3}});
            obj.decodeLayer.params{1} = obj.get_encode_params_transposed();
         else
            splitIdx = length(obj.encodeLayer.params);
            obj.encodeLayer.increment_params(delta_params(1:splitIdx));
            obj.decodeLayer.increment_params(delta_params(splitIdx+1:end));
         end
      end
      
      function gather(obj)
         obj.encodeLayer.gather();
         obj.decodeLayer.gather();
      end
      
      function push_to_GPU(obj)
         obj.encodeLayer.push_to_GPU();
         obj.decodeLayer.push_to_GPU();
      end
      
      function reset(obj)
         obj.encodeLayer.init_params();
         obj.decodeLayer.init_params();
         if obj.isTiedWeights 
            obj.decodeLayer.params{1} = obj.get_encode_params_transposed();
         end
      end
      
      function pTrans = get_encode_params_transposed(obj)
         if ndims(obj.encodeLayer.params{1}) <= 2
            pTrans = obj.encodeLayer.params{1}';
         else
            pTrans = permute(obj.encodeLayer.params{1}, [2, 1, 3]);
         end
      end
      
      function objCopy = copy(obj)
         objCopy = AutoEncoder();
         % Handle properties
         objCopy.encodeLayer = obj.encodeLayer.copy();
         objCopy.decodeLayer = obj.decodeLayer.copy();
         
         % Value properties
         objCopy.isTiedWeights = obj.isTiedWeights;
         objCopy.dropout = obj.dropout;
         objCopy.isDropout = obj.isDropout;
         objCopy.gpuState = obj.gpuState;
      end
   end
   
end

