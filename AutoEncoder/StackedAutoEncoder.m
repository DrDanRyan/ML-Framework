classdef StackedAutoEncoder < handle
   % This defines the interface for an StackedAutoEncoder

   methods 
      pretrain
      [grad, output] = gradient(obj, x)
      xRep = encode(x)
      xRecon = decode(xRep)
      loss = compute_loss(x, xRecon)
      increment_params(obj, delta_params)
      push_to_GPU(obj)
      gather(obj)
      reset(obj)
      W = get_encoding_matrix(obj)
      V = get_decoding_matrix(obj)
   end
   
end

