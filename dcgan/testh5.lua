

require 'xlua'
require 'optim'
require 'hdf5'

local function readh5(path)
      	local myFile = hdf5.open(path, 'r')
        --print(myFile)
	local data = myFile:read('feat'):all()
   
	myFile:close()
      return data
end


local path = '/home/student/icml2016/eeg_h5files/n02106662_1152.h5'
local feat = readh5(path)
print(feat)

