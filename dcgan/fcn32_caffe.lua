require 'caffe'
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'

n_class = 33

model = nn.Sequential()
--model:add(caffe.Net('/home/student/caffe_orig/caffe-master/VOC/deploy_32.prototxt', '/home/student/caffe_orig/caffe-master/VOC/snapshots/deconv_voc_fc32_train50_21cls_iter_100000.caffemodel', 'test'))
model:add(caffe.Net('/home/student/caffe_orig/caffe-master/models/VGG_ILSVRC_16_layers_deploy_conv.prototxt', '/home/student/caffe_orig/caffe-master/models/VGG_ILSVRC_16_layers_conv.caffemodel', 'test'))
--model:add(nn.Linear(1000,1))
--   score_fr=L.Convolution2D(4096, self.n_class, 1, stride=1, pad=0),

--            upscore2=L.Deconvolution2D(self.n_class, self.n_class, 4,
--                                       stride=2, pad=0, use_cudnn=False),
--            upscore8=L.Deconvolution2D(self.n_class, self.n_class, 16,
--                                       stride=8, pad=0, use_cudnn=False),

--            score_pool3 = L.Convolution2D(256, self.n_class, 1, stride=1, pad=0),
--            score_pool4 = L.Convolution2D(512, self.n_class, 1, stride=1, pad=0),
--            upscore_pool4= L.Deconvolution2D(self.n_class, self.n_class, 4,
--                                            stride=2, pad=0, use_cudnn=False),
-- # score_pool3
--        h = self.score_pool3(pool3)
--        score_pool3 = h  # 1/8

--        # score_pool4
--        h = self.score_pool4(pool4)
--        score_pool4 = h  # 1/16

--        # upscore2
--        h = self.upscore2(score_fr)
--        upscore2 = h  # 1/16

--        # score_pool4c
--        h = score_pool4[:, :,
--            5:5 + upscore2.data.shape[2], 5:5 + upscore2.data.shape[3]]
--        score_pool4c = h  # 1/16

--        # fuse_pool4
--        h = upscore2 + score_pool4c
--        fuse_pool4 = h  # 1/16

--        # upscore_pool4
--        h = self.upscore_pool4(fuse_pool4)
--        upscore_pool4 = h  # 1/8

--        # score_pool4c
--        h = score_pool3[:, :,
--            9:9 + upscore_pool4.data.shape[2],
--            9:9 + upscore_pool4.data.shape[3]]
--        score_pool3c = h  # 1/8

--        # fuse_pool3
--        h = upscore_pool4+ score_pool3c
--        fuse_pool3 = h  # 1/8

--        # upscore8
--        h = self.upscore8(fuse_pool3)
--        upscore8 = h  # 1/1

--        # score
--        h = upscore8[:, :, 31:31 + x.data.shape[2], 31:31 + x.data.shape[3]]
--        self.score = h  # 1/1


--        # self.blob_class = self.classi(h)
--        # self.probs = F.softmax(self.blob_class)
--        g_z = F.argmax(h, axis=1)
model.add(nn.SpatialConvolution(4096,n_class,1,1,1,1,0,0))
model.add(nn.SpatialConvolution(n_class,n_class,1,1,1,1,0,0))
