classdef AutoEncoder < AutoEncoderInterface
   % Basic AutoEncoder
   
   properties
      encodeLayer % a HiddenLayer object that functions as the encoding layer
      isTiedWeights % a boolean indicating if the params in encodeLayer and 
                    % decodeLayer are shared; assumes encodeLayer.params = {W, b} 
                    % when isTiedWeights is true
      gpuState
      encodeGradSize % size of cell array returned by enocdeLayer.backprop
      
      isImputeValues % if true, NaN values will be imputed by a recursive
                     % feed-forward scheme (keeping non-NaN values clamped)
      imputeTol % a tolerance for the max absolute difference for imputed values
                % after one cycle through the AutoEncoder
   end
   
   properties (Dependent)
      decodeLayer % a OutputLayer object that functions as the decoding layer 
                  % and loss function
   end
   
   methods
      function obj = AutoEncoder(varargin)
         p = inputParser;
         p.KeepUnmatched = true;
         p.addParamValue('isTiedWeights', false);
         p.addParamValue('isImputeValues', false);
         p.addParamValue('imputeTol', 1e-5);
         p.addParamValue('gpu', []);
         parse(p, varargin{:});

         obj.isTiedWeights = p.Results.isTiedWeights;
         obj.isImputeValues = p.Results.isImputeValues;
         obj.imputeTol = p.Results.imputeTol;
         
         if isempty(p.Results.gpu)
            obj.gpuState = GPUState();
         else
            obj.gpuState = GPUState(p.Results.gpu);
         end
      end
      
      function set.decodeLayer(obj, decodeLayer)
         % Ties the weights of the decodeLayer to the encodeLayer if
         % isTiedWeights is true.
         obj.decodeLayer = decodeLayer;
         if obj.isTiedWeights
            obj.decodeLayer.params{1} = obj.get_encode_params_transposed();
         end
      end
      
      function [grad, xRecon] = gradient(obj, batch)
         if obj.isImputeValues
            x = obj.impute_values(batch{1});
         else
            x = batch{1};
         end
         
         h = obj.encodeLayer.feed_forward(x, true);
         [decodeGrad, dLdh, xRecon] = obj.decodeLayer.backprop(h, x);
         encodeGrad = obj.encodeLayer.backprop(x, h, dLdh);

         if obj.isTiedWeights
            grad = obj.tied_weights_grad(encodeGrad, decodeGrad);
         else
            obj.encodeGradSize = length(encodeGrad);
            grad = [encodeGrad, decodeGrad];
         end
      end
      
      function grad = tied_weights_grad(~, encodeGrad, decodeGrad)
         if ndims(encodeGrad{1}) <= 2
               grad = {encodeGrad{1}+decodeGrad{1}', encodeGrad{2}, ...
                       decodeGrad{2}};
         else
               grad = {encodeGrad{1}+permute(decodeGrad{1}, [2, 1, 3]), ...
                        encodeGrad{2}, decodeGrad{2}};
         end
      end
      
      function loss = compute_loss(obj, batch)
         if obj.isImputeValues
            x = obj.impute_values(batch{1});
         else
            x = batch{1};
         end
         xRecon = obj.output(x);
         loss = obj.compute_loss_from_output(xRecon, x);
      end
      
      function loss = compute_loss_from_output(obj, xRecon, x)
         loss = obj.decodeLayer.compute_loss(xRecon, x);
      end
      
      function h = encode(obj, x)
         if obj.isImputeValues
            x = obj.impute_values(x);
         end
         h = obj.encodeLayer.feed_forward(x);
      end
      
      function xRecon = output(obj, x)
         if obj.isImputeValues
            x = obj.impute_values(x);
         end
         h = obj.encodeLayer.feed_forward(x);
         xRecon = obj.decodeLayer.feed_forward(h);
      end
      
      function increment_params(obj, delta)
         if obj.isTiedWeights
            obj.encodeLayer.increment_params(delta(1:2));
            obj.decodeLayer.increment_params({0, delta{3}});
            obj.decodeLayer.params{1} = obj.get_encode_params_transposed();
         else
            obj.encodeLayer.increment_params(delta(1:obj.encodeGradSize));
            obj.decodeLayer.increment_params(delta(obj.encodeGradSize+1:end));
         end
      end
      
      function gather(obj)
         obj.encodeLayer.gather();
         obj.decodeLayer.gather();
         obj.gpuState.isGPU = false;
      end
      
      function push_to_GPU(obj)
         obj.encodeLayer.push_to_GPU();
         obj.decodeLayer.push_to_GPU();
         obj.gpuState.isGPU = true;
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
      
      function x = impute_values(obj, x)
         nanIdx = isnan(x);
         if isempty(nanIdx)
            return
         end
         
         absDiff = Inf;
         x(nanIdx) = 0;
         while absDiff > obj.imputeTol
            h = obj.encodeLayer.feed_forward(x);
            xRecon = obj.decodeLayer.feed_forward(h);
            absDiff = max(abs(x(nanIdx) - xRecon(nanIdx)));
            x(nanIdx) = xRecon(nanIdx);
         end
      end
      
      function objCopy = copy(obj)
         objCopy = AutoEncoder();
         % Handle properties
         objCopy.encodeLayer = obj.encodeLayer.copy();
         objCopy.decodeLayer = obj.decodeLayer.copy();
         
         % Value properties
         objCopy.isTiedWeights = obj.isTiedWeights;
         objCopy.isImputeValues = obj.isImputeValues;
         objCopy.imputeTol = obj.imputeTol;
         objCopy.gpuState = obj.gpuState;
      end
   end
   
end

