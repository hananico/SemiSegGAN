require 'loadcaffe'
require 'xlua'
require 'optim'
 

-- to train lenet network please follow the steps
-- provided in CAFFE_DIR/examples/mnist
prototxt = '/home/student/caffe_orig/caffe-master/models/VGG_ILSVRC_16_layers_deploy_conv.prototxt'
binary = '/home/student/caffe_orig/caffe-master/models/VGG_ILSVRC_16_layers_conv.caffemodel'

-- this will load the network and print it's structure
net = loadcaffe.load(prototxt, binary)

input = torch.randn(10,3,224,224):double()
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


