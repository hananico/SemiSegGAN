
require 'xlua'
require 'optim'
require 'hdf5'


local file = io.open("/home/student/caffe_orig/caffe-master/EEG/EEG_data/eeg_features_filter_average.txt")
if file then
local database = { }
 i=0
    for line in file:lines() do
        local f, c, fe = line:gmatch '(%S+)%s+(%S+)%s+(%S+)'
      --  local subj, file, class, feat = unpack(line:split(",")) --unpack turns a table like the one given (if you use the recommended version) into a bunch of separate variables
        --do something with that data
    table.insert(database, {file =f, class = c, feat = fe })
    print(line)
    print(f)
    print(i)
    i = i +1
    end
    
else
end
