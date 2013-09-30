classdef AutoEncoder < handle
   % This abstract class defines the AutoEncoder interface.
   
   properties (Abstract)
      encodeLayer % a HiddenLayer object that functions as the encoding layer
      decodeLayer % a HiddenLayer object that functions as the decoding layer
      lossFunction % a LossFunction object that compute the loss and loss derivative
   end
   
   methods (Abstract)
      grad = gradient(obj, x)
      update_params(obj, delta_params)
      gather(obj)
      push_to_GPU(obj)
      reset(obj)
   end
   
end

