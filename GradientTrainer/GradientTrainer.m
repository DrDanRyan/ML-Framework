classdef GradientTrainer < handle
   
   properties
      dataManager % implements DataManager interface
      model % implements Model interface
      reporter % implements Reporter interface
      stepCalculator % implements StepCalculator interface
      trainingSchedule % implements TrainingSchedule interface
   end
   
   methods
      function train(obj)
         isContinue = true;
         while isContinue
            [x, t, isEndOfEpoch] = obj.dataManager.next_batch();
            obj.stepCalculator.take_step(x, t, obj.model, obj.trainingSchedule.params);
            
            if isEndOfEpoch
               trainingLoss = ...
                  obj.model.compute_loss(obj.model.output(obj.dataManager.trainingInputs), ...
                                         obj.dataManager.trainingTargets);
               validationLoss = ...
                  obj.model.compute_loss(obj.model.output(obj.dataManager.validationInputs), ...
                                         obj.dataManager.validationTargets);
                                                    
               obj.reporter.update(trainingLoss, validationLoss);
               isContinue = obj.trainingSchedule.update(obj, trainingLoss, validationLoss);
            end             
         end
      end
      
      function objCopy = copy(obj)
         objCopy = GradientTrainer();
         objCopy.dataManager = copy(obj.dataManager);
         objCopy.model = copy(obj.model);
         objCopy.reporter = copy(obj.reporter);
         objCopy.stepCalculator = copy(obj.stepCalculator);
         objCopy.trainingSchedule = copy(obj.trainingSchedule);
      end
      
      function reset(obj)
         obj.reporter.reset();
         obj.stepCalculator.reset();
         obj.trainingSchedule.reset();
         obj.dataManager.reset();
         obj.model.reset();
      end
   end      
end

