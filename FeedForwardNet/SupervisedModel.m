classdef SupervisedModel < handle
   % This defines the Model interface
   
   methods (Abstract)
      [grad, output] = gradient(obj, x, t) % obj: Model, x: input data, t: targets
      y = output(obj, x)
      loss = compute_loss(obj, y, t) % y: output, t: target
      increment_params(obj, delta)
      gather(obj) % brings data from GPU memory to main memory
      push_to_GPU(obj) % push data from main memory onto GPU memory
      objCopy = copy(obj) % make an identical copy of the current model
      reset(obj) % reinitialize all the model parameters to initial random states
   end
   
end

