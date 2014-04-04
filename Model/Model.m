classdef Model < handle
   % This defines the Model interface. This interface is sufficient to work with
   % all currently implemented StepCalculator objects and ProgressMonitor
   % objects.
   
   methods (Abstract)
      [grad, output] = gradient(obj, batch)
      y = output(obj, x)
      loss = compute_loss(batch)
      loss = compute_loss_from_output(obj, y, t) % y: output, t: target
      
      % increment model parameters according to step calculated by 
      % StepCalculator object
      increment_params(obj, delta) 
      
      gather(obj) % brings data from GPU memory to main memory
      push_to_GPU(obj) % push data from main memory onto GPU memory
      objCopy = copy(obj) % make an identical copy of the current model
      
      % reinitialize all the model parameters to initial random states
      reset(obj) 
   end
   
end

