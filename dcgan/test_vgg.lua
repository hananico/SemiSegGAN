require 'loadcaffe'
require 'xlua'
require 'optim'
require 'hdf5'

-- to train lenet network please follow the steps
-- provided in CAFFE_DIR/examples/mnist
prototxt = '/home/student/caffe_orig/caffe-master/models/VGG_ILSVRC_16_layers_deploy_conv.prototxt'
binary = '/home/student/caffe_orig/caffe-master/models/VGG_ILSVRC_16_layers_conv.caffemodel'

s ='images/2011_003025.png'  
i = string.find(s, "/")
j = string.find(s, ".png")

    print(i, j)                     
    print(string.sub(s, i+1, j-1)) 
exit(0)

local myFile = hdf5.open('/home/student/icml2016/VGG_fc7/2007_000039.h5', 'r')
local data = myFile:read('fc7'):all()
myFile:close()
print(data:size())
	
-- this will load the network and print it's structure
net = loadcaffe.load(prototxt, binary)

input = torch.randn(1,3,224,224):double()

output = net:forward(input)
gradInput = net:backward(input, output)
local conv_nodes = net:findModules('nn.SpatialConvolution')
print(conv_nodes[#conv_nodes-1].output)
--print(net:listModules())
--
--for i = 1, #conv_nodes do
 -- print(i)
 -- print(conv_nodes[i].output:size())
--end


