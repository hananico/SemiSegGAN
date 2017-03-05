require 'torch'
require 'nngraph'
require 'cunn'
-- require 'optim'
require 'image'
-- require 'pl'
-- require 'paths'

-- require 'caffe'
require 'nn'
require 'loadcaffe'

-- ok, disp = pcall(require, 'display')
-- if not ok then print('display not found. unable to plot') end

n_class = 33
loadSize = {3, 256}

local createNetwork = function(width, height)
	fcn32 = nn.Sequential()

	dx_I = nn.Identity()()

	-- 3: input size (rgb)
	-- 64: num filter 
	-- 3x3: kernel size
	-- 1x1 stride
	-- 100x100 padding

	dc1_1 = nn.SpatialConvolution(3,64,3,3,1,1,100,100)(dx_I)
	dr1_1 = nn.ReLU(true)(dc1_1)
	dc1_2 = nn.SpatialConvolution(64,64,3,3,1,1,1,1)(dr1_1)
	dr1_2 = nn.ReLU(true)(dc1_2)
	dp1 = nn.SpatialMaxPooling(2,2,2,2)(dr1_2)

	dc2_1= nn.SpatialConvolution(64,128,3,3,1,1,1,1)(dp1)
	dr2_1 =nn.ReLU(true)(dc2_1)
	dc2_2 =nn.SpatialConvolution(128,128,3,3,1,1,1,1)(dr2_1)
	dr2_2 =nn.ReLU(true)(dc2_2)
	dp2 = nn.SpatialMaxPooling(2,2,2,2)(dr2_2)

	dc3_1 =nn.SpatialConvolution(128,256,3,3,1,1,1,1)(dp2)
	dr3_1 =nn.ReLU(true)(dc3_1)
	dc3_2 = nn.SpatialConvolution(256,256,3,3,1,1,1,1)(dr3_1)
	dr3_2 =nn.ReLU(true)(dc3_2)
	dc3_3 = nn.SpatialConvolution(256,256,3,3,1,1,1,1)(dr3_2)
	dr3_3= nn.ReLU(true)(dc3_3)
	dp3 =nn.SpatialMaxPooling(2,2,2,2)(dr3_3)

	dc4_1 =nn.SpatialConvolution(256,512,3,3,1,1,1,1)(dp3)
	dr4_1 =nn.ReLU(true)(dc4_1)
	dc4_2 = nn.SpatialConvolution(512,512,3,3,1,1,1,1)(dr4_1)
	dr4_2 =nn.ReLU(true)(dc4_2)
	dc4_3 = nn.SpatialConvolution(512,512,3,3,1,1,1,1)(dr4_2)
	dr4_3= nn.ReLU(true)(dc4_3)
	dp4 =nn.SpatialMaxPooling(2,2,2,2)(dr4_3)


	dc5_1 =nn.SpatialConvolution(512,512,3,3,1,1,1,1)(dp4)
	dr5_1 =nn.ReLU(true)(dc5_1)
	dc5_2 = nn.SpatialConvolution(512,512,3,3,1,1,1,1)(dr5_1)
	dr5_2 =nn.ReLU(true)(dc5_2)
	dc5_3 = nn.SpatialConvolution(512,512,3,3,1,1,1,1)(dr5_2)
	dr5_3= nn.ReLU(true)(dc5_3)
	dp5 =nn.SpatialMaxPooling(2,2,2,2)(dr5_3)


	--fc6
	df6 =nn.SpatialConvolution(512,4096,7,7,1,1,0,0)(dp5)
	dr6 = nn.ReLU(true)(df6)
	ddr6 =nn.Dropout()(dr6)

	--fc7
	df7 = nn.SpatialConvolution(4096,4096,1,1,1,1,0,0)(ddr6)
	dr7 =nn.ReLU(true)(df7)
	ddr7 = nn.Dropout()(dr7)

	--score_fr
	dsf =nn.SpatialConvolution(4096,n_class,1,1,1,1,0,0)(ddr7)

	--score_pool3
	--dsf_p3 =nn.SpatialConvolution(256,n_class,1,1,1,1,0,0)(dp3)


	--score_pool4
	--dsf_p4 =nn.SpatialConvolution(512,n_class,1,1,1,1,0,0)(dp4)

	-- upscore
	dup = nn.SpatialFullConvolution(n_class,n_class,64,64,32,32,0,0)(dsf)


	print ("width:"  .. width)
	print ("height:"  .. height)
	-- print ("dup:"  .. dup:size())
	
	--"Crop" TODO: ::size

-- TODO: -1 might not be correct, old value was width and heigth

	dcrp1 = nn.Narrow(3,19,256)(dup)
	dcrp2= nn.Narrow(4,19,256)(dcrp1)

	net = nn.gModule({dx_I},{dcrp2})
	
	return net
	
	--fuse_pool4
	-- dfusp4= nn.JoinTable(...){dlcrp4,dup2}
	--  Fused1 = nn.CAddTable()({L14c2,PS1})
end

local function loadImage(path, depth)
	local input
	
	print("Loading image: " .. path ..", depth:"..depth)

	if depth == 1 then
		input= image.load(path, depth, 'byte')
	else
		input= image.load(path, depth, 'float')
	end

	if filter255 then
		input[input:gt(0)] = 255
	end

	input = image.scale(input, loadSize[2], loadSize[2])

	return input
end

local function loadCaffeModel()
	--model:add(caffe.Net('/home/student/caffe_orig/caffe-master/VOC/deploy_32.prototxt', '/home/student/caffe_orig/caffe-master/VOC/snapshots/deconv_voc_fc32_train50_21cls_iter_100000.caffemodel', 'test'))
	--caffemodel:add(caffe.Net('/home/student/caffe_orig/caffe-master/models/VGG_ILSVRC_16_layers_deploy_conv.prototxt', '/home/student/caffe_orig/caffe-master/models/VGG_ILSVRC_16_layers_conv.caffemodel', 'test'))
	
	-- prototxt = '/home/student/caffe_orig/caffe-master/models/VGG_ILSVRC_16_layers_deploy_conv.prototxt'
	-- caffemodel_path = '/home/student/caffe_orig/caffe-master/models/VGG_ILSVRC_16_layers_conv.caffemodel'
	
	folder = '/home/student/caffe_orig/caffe-master/fcn.berkeleyvision.org-master/siftflow-fcn32s/'
	prototxt =  '/home/student/icml2016//deploy_fcn32.prototxt'
	caffemodel_path =folder .. "siftflow-fcn32s-heavy.caffemodel"
	print("Loading caffe, proto: " .. prototxt)
	print("Loading caffe, model: " .. caffemodel_path)
	local caffemodel = loadcaffe.load(prototxt,caffemodel_path,'nn')
	
	return caffemodel
end


batch_size = 1
local data_img_seq1 = torch.zeros(batch_size, 3, loadSize[2], loadSize[2])
local data_lbl_seq1 = torch.zeros(batch_size, 1, loadSize[2], loadSize[2])

input_image = loadImage('/home/student/caffe_orig/caffe-master/data/VOC2012_SEG_AUG/images/2008_008606.png', 3)
label = loadImage('/home/student/caffe_orig/caffe-master/data/VOC2012_SEG_AUG/segmentations/2008_008606.png', 1)


data_img_seq1:copy(input_image)
data_lbl_seq1:copy(label)

net = createNetwork(loadSize[2], loadSize[2])
output = net:forward(data_img_seq1, data_lbl_seq1)
n_class = 33
print(output:size())

caffemodel = loadCaffeModel()
print(caffemodel:size())

local torch_parameters = net:getParameters()
print (torch_parameters)
local caffeparameters = caffemodel:getParameters()
print (caffeparameters)
torch_parameters:copy(caffeparameters)

