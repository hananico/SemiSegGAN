require 'torch'
require 'nngraph'
require 'cunn'
require 'optim'
require 'image'
require 'pl'
require 'paths'
require 'caffe'
require 'nn'

image_utils = require 'image'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end



n_class = 33

fcn32 = nn.Sequential()

require 'nn'

dx_I = nn.Identity()()

dc1_1 = nn.SpatialConvolution(3,64,3,3,1,1,100,100)(dx_I)
dr1_1 = nn.ReLU(true)(dc1_1)
dc1_2 = nn.SpatialConvolution(64,64,3,3,1,1,1,1)(dr_1_1)
dr1_2 = nn.ReLU(true)(dc1_2)
dp1 = SpatialMaxPooling(2,2,2,2)(dr1_2)

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


dc5_1 =nn.SpatialConvolution(512,512,3,3,1,1,1,1)(dp5)
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
dsf =nn.SpatialConvolution(4096,n_class,1,1,1,1,0,0)(df7)

--score_pool3
--dsf_p3 =nn.SpatialConvolution(256,n_class,1,1,1,1,0,0)(dp3)


--score_pool4
--dsf_p4 =nn.SpatialConvolution(512,n_class,1,1,1,1,0,0)(dp4)

-- upscore
dup = nn.SpatialFullConvolution(n_class,n_class,64,64,32,32,0,0)(dsf)


--"Crop"
dcrp1 = nn.Narrow(2,1,dx_I::size(1))(dup)
dcrp2= nn.Narrow(3,1,dx_I::size(1))(dcrp1)

net = nn.gModule({dx_I},{dcrp2})
--fuse_pool4
-- dfusp4= nn.JoinTable(...){dlcrp4,dup2}
--  Fused1 = nn.CAddTable()({L14c2,PS1})

