classdef Conv2DFilterReporter < Reporter
   % As of now only works for filters with single input channel (e.g. first
   % layer filters on grayscale images)
   
   properties
      convLayer
      rows
      cols
      consoleReporter
   end
   
   methods
      function obj = Conv2DFilterReporter(convLayer, rows, cols)
         obj.convLayer = convLayer;
         obj.rows = rows;
         obj.cols = cols;
         obj.consoleReporter = ConsoleReporter();
         obj.init_figure();         
      end
      
      function init_figure(obj)
         figure();
         for i = 1:obj.rows
            for j = 1:obj.cols
               subtightplot(obj.rows, obj.cols, obj.cols*(i-1) + j, [.005, .005]);
               [~, ~, ~, fRows, fCols] = size(obj.convLayer.params{1});
               image(zeros(fRows, fCols), 'CDataMapping', 'scaled');
               colormap('gray');
               axis off tight
            end
         end
         drawnow;
      end
      
      function report(obj, progressMonitor, ~)
         for i = 1:obj.rows
            for j = 1:obj.cols
               idx = obj.cols*(i-1) + j;
               subtightplot(obj.rows, obj.cols, idx, [.005, .005]);
               f = squeeze(obj.convLayer.params{1}(idx,:,:,:,:));
               M = gather(max(abs(f(:))));
               image(f, 'CDataMapping', 'scaled');
               set(gca, 'CLim', [-M-1e-4, M+1e-4]);
               axis off tight
            end
         end
         drawnow;
         obj.consoleReporter.report(progressMonitor, []);
      end
      
      function reset(obj)
         close;
         obj.init_figure();
      end
      
   end
end

