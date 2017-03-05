--[[

Generic training script for GAN, GAN-CLS, GAN-INT, GAN-CLS-INT.

--]]
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
--require 'caffe'
require 'cudnn'




local ffi = require 'ffi'
require 'loadcaffe'
local C = loadcaffe.C

opt = {
   numCaption = 1,
   large = 0,
   save_every = 1,
   print_every = 1,
   dataset = 'voc',
   img_dir = '/home/student/caffe_orig/caffe-master/data/VOC2012_SEG_AUG',
   feat_dir='/home/student/icml2016/VGG_fc7' ,
   cls_weight = 0.05,
   filenames = '',
   data_root = '/home/student/caffe_orig/caffe-master/data/VOC2012_SEG_AUG',
   checkpoint_dir = '/home/student/icml2016/checkpoints/',
   batchset = './batch_voc_train.txt' ,
   batchSize = 15,  -- 64 -> 10
   loadSize = 256,  -- 256 -> 64
   fineSize = 256,  -- 256 -> 64
   txtSize = 12288,         -- #  come from caffe, fc7
   nt = 1024,               -- #  of dim for text features.
   nz = 100,               -- #  of dim for Z
   ngf = 128,              -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 0,           -- #  of data loading threads to use
   niter = 1000,             -- #  of iter at starting learning rate
   lr = 0.000001,            -- initial learning rate for adam
   lr_decay = 0.5,            -- initial learning rate for adam
   decay_every = 10,
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   gpu_index =1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'semiGAN',
   noise = 'normal',       -- uniform / normal
   init_g = '',
   init_d = '',
   use_cudnn = 1,
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

if opt.gpu > 0 then
   ok, cunn = pcall(require, 'cunn')
   print("Nasim 3, ok: " .. tostring(ok))

   ok2, cutorch = pcall(require, 'cutorch')
   print("Nasim 4, ok2: " .. tostring(ok2))

   print("GPU count: " .. tostring(cutorch.getDeviceCount()))
   print("GPU current active: " .. tostring(cutorch.getDevice()))

   print("using GPU:: " .. opt.gpu_index)

   cutorch.setDevice(opt.gpu_index)
end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')
print("start")
-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------

local function loadcaffe_load(prototxt_name, binary_name, backend, last_layer)
   local backend = backend or 'nn'
   local handle = ffi.new('void*[1]')

   -- loads caffe model in memory and keeps handle to it in ffi
   local old_val = handle[1]
   C.loadBinary(handle, prototxt_name, binary_name)
   if old_val == handle[1] then return end

   -- transforms caffe prototxt to torch lua file model description and
   -- writes to a script file
   local lua_name = prototxt_name..'.lua'

   -- C.loadBinary creates a .lua source file that builds up a table
   -- containing the layers of the network. As a horrible dirty hack,
   -- we'll modify this file when backend "nn-cpu" is requested by
   -- doing the following:
   --
   -- (1) Delete the lines that import cunn and inn, which are always
   --     at lines 2 and 4
   local model = nil
   if backend == 'nn-cpu' then
      C.convertProtoToLua(handle, lua_name, 'nn')
      local lua_name_cpu = prototxt_name..'.cpu.lua'
      local fin = assert(io.open(lua_name), 'r')
      local fout = assert(io.open(lua_name_cpu, 'w'))
      local line_num = 1
      while true do
         local line = fin:read('*line')
         if line == nil then break end
         fout:write(line, '\n')
         line_num = line_num + 1
      end
      fin:close()
      fout:close()
      model = dofile(lua_name_cpu)
   else
      C.convertProtoToLua(handle, lua_name, backend)
      model = dofile(lua_name)
   end

   -- goes over the list, copying weights from caffe blobs to torch tensor
   local net = nn.Sequential()
   local list_modules = model
   for i,item in ipairs(list_modules) do
      item[2].name = item[1]
      if item[2].weight then
         local w = torch.FloatTensor()
         local bias = torch.FloatTensor()
         C.loadModule(handle, item[1], w:cdata(), bias:cdata())
         if backend == 'ccn2' then
            w = w:permute(2,3,4,1)
         end
         item[2].weight:copy(w)
         item[2].bias:copy(bias)
      end
      net:add(item[2])
      if i == last_layer then
         break
      end
   end
   C.destroyBinary(handle)

   if backend == 'cudnn' or backend == 'ccn2' then
      net:cuda()
   end

   return net
end

-- quiver plots
local function meshgrid(size1,size2)
   xx = torch.linspace(-3,3,size1)
   yy = torch.linspace(-3,3,size2)
	
   local X = torch.repeatTensor(xx, yy:size(1),1)
   local Y = torch.repeatTensor(yy:view(-1,1), 1, xx:size(1))
   return X, Y
end

local function ogrid(size1, size2)
	xx = torch.Tensor(size1, 1)
	for i=1,size1 do xx[i][1] = i-1 end
	yy = xx:transpose(1,2)
	return xx, yy
end
	
	
local function upsample_filt(size)
   factor = math.floor((size + 1) / 2)
   print ("factor: " .. factor)
   if size % 2 == 1 then
      center = factor - 1
   else
      center = factor - 0.5
   end
   print ("center: " .. center)
   xx, yy = ogrid(size, size)
   print ("here")
   print (xx)
   print (yy)
   return (1 - torch.abs(xx - center) / factor) * (1 - torch.abs(yy - center) / factor)
end
----------------------------------------------------------------------------

local function weights_init_up(m)
   local name = torch.type(m)
   if name:find('SpatialFullConvolution') then
	  m_size = m.output:size()
	  width = m_size[1]
	  height = m_size[2]
	  if (weight ~= height) then
		error ("filters need to be square w: " .. weight .. " , height: " .. height)
      end

  	  filt = upsample_filt(height)
	  print ("filt:")
	  print (filt)
	  print ("m.weight:")
	  print (m.weight)
	  print ("m.weight:size:")
	  print (m.weight:size())	
      m.weight = filt
      m.bias:fill(0)
   end
end

--local function  interp(net, layers)
--    for l in layers
--        i_channel, o_channel, height, width = net.params[l][0].data.shape
--        if i_channel != o_channel and o_channel != 1:
--            print 'input + output channels need to be the same or |output|1 == 1'
--            raise
--        if height != width:
--            print 'filters need to be square'
--            raise
--        filt = upsample_filt(height)
--        net.params[layers][0].data[range(i_channel), range(o_channel), :, :] = filt

   n_class = 21

   local nc = 1 -- Number of channels is 1 for segment
   local nz = opt.nz
   local ndf = opt.ndf
   local ngf = opt.ngf
   local real_label = 1
   local fake_label = 0

   local SpatialBatchNormalization = nn.SpatialBatchNormalization
   local SpatialConvolution = nn.SpatialConvolution
   local SpatialFullConvolution = nn.SpatialFullConvolution



--if opt.init_d == '' then
--  convD = nn.Sequential()
--  -- input is (nc) x 64 x 64
--  convD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
--  convD:add(nn.LeakyReLU(0.2, true))
--  -- state size: (ndf) x 32 x 32
--  convD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
--  convD:add(SpatialBatchNormalization(ndf * 2))
--  convD:add(nn.LeakyReLU(0.2, true))

--  -- state size: (ndf*2) x 16 x 16
--  convD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
--  convD:add(SpatialBatchNormalization(ndf * 4))
--  convD:add(nn.LeakyReLU(0.2, true))

--  -- state size: (ndf*4) x 8 x 8
--  convD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
--  convD:add(SpatialBatchNormalization(ndf * 8))

--    -- state size: (ndf*8) x 4 x 4
--    local conc = nn.ConcatTable()
--    local conv = nn.Sequential()
--    conv:add(SpatialConvolution(ndf * 8, ndf * 2, 1, 1, 1, 1, 0, 0))
--    conv:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
--    conv:add(SpatialConvolution(ndf * 2, ndf * 2, 3, 3, 1, 1, 1, 1))
--    conv:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
--    conv:add(SpatialConvolution(ndf * 2, ndf * 8, 3, 3, 1, 1, 1, 1))
--    conv:add(SpatialBatchNormalization(ndf * 8))
--    conc:add(nn.Identity())
--    conc:add(skipD)
--    convD:add(conc)
--    convD:add(nn.CAddTable())

--  convD:add(nn.LeakyReLU(0.2, true))

--  local fcD = nn.Sequential()
--  fcD:add(nn.Linear(opt.txtSize,opt.nt))
--  fcD:add(nn.LeakyReLU(0.2,true))
--  fcD:add(nn.Replicate(4,3))
--  fcD:add(nn.Replicate(4,4))
--  netD = nn.Sequential()
--  pt = nn.ParallelTable()
--  pt:add(convD)
--  pt:add(fcD)
--  netD:add(pt)
--  netD:add(nn.JoinTable(2))
--  -- state size: (ndf*8 + 128) x 4 x 4
--  netD:add(SpatialConvolution(ndf * 8 + opt.nt, ndf * 8, 1, 1))
--  netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
--  netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
--  netD:add(nn.Sigmoid())
--  -- state size: 1 x 1 x 1
--  netD:add(nn.View(1):setNumInputDims(3))
--  -- state size: 1
--  netD:apply(weights_init)
--else
--  netD = torch.load(opt.init_d)
--end
   local createNetwork = function(n_class,width, height)
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



      -- print ("dup:"  .. dup:size())

      --"Crop" TODO: ::size

-- TODO: -1 might not be correct, old value was width and heigth

      dcrp1 = nn.Narrow(3,19,256)(dup)
      dcrp2= nn.Narrow(4,19,256)(dcrp1)



      net = nn.gModule({dx_I},{dcrp2})

--   print ("width:"  .. width)
--   print ("height:"  .. height)
      return net

      --fuse_pool4
      -- dfusp4= nn.JoinTable(...){dlcrp4,dup2}
      --  Fused1 = nn.CAddTable()({L14c2,PS1})
   end

   folder = '/home/student/caffe_orig/caffe-master/VOC/'
   prototxt =  folder .. '/deploy_32_4lua.prototxt'
   prototex = '/home/student/caffe_orig/caffe-master/fcn.berkeleyvision.org-master/voc-fcn32s/train.prototxt'

--caffemodel_path = folder .. '/snapshots/voc_fcn8_train1400_cls22_iter_10000.caffemodel'
   caffemodel_path ='/home/student/caffe_orig/caffe-master/fcn.berkeleyvision.org-master/voc-fcn32s/fcn32s-heavy-pascal.caffemodel'
--caffemodel = caffe.Net(prototxt, caffemodel_path,'train')
   print("NASIM 0000")
   caffemodel = loadcaffe.load(prototxt,caffemodel_path,'nn')
   print("NASIM 1111")
   print(caffemodel)
   print("NASIM 2222")
-- caffemodel = loadcaffe_load(prototxt,caffemodel_path,'nn', 'score')

--netD = nn.Sequential()

--netD:add(caffemodel)
   local upscore =  nn.SpatialFullConvolution(21, 21, 64, 64, 32, 32, 0, 0)
   local dcrp1 = nn.Narrow(3,19,256)
   local dcrp2= nn.Narrow(4,19,256)
   local softMaxLayer = nn.LogSoftMax()


--netD:add(softMaxLayer)

-- caffemodel:add(softMaxLayer)

   caffemodel:insert(upscore)
   caffemodel:insert(dcrp1)
   caffemodel:insert(dcrp2)
--caffemodel:insert(softMaxLayer)

	caffemodel:apply(weights_init_up)
   print ("NASIM NASIM NASIM")

-- netD= createNetwork(n_class, opt.loadSize,opt.loadSize)
   netD= caffemodel


   print(netD)

--torch.save(opt.checkpoint_dir .. '/' ..  'caffemodel_net_D_fcn32.t7', caffemodel)


--local output = model:forward(img:cuda()):squeeze(1)
   local criterion = cudnn.SpatialCrossEntropyCriterion()

   optimStateD = {
      learningRate = opt.lr,
      beta1 = opt.beta1,
   }
----------------------------------------------------------------------------
   local input_segment = torch.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
   local input_image = torch.FloatTensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
--local input_image = torch.Tensor(opt.batchSize, opt.txtSize)

   local noise = torch.Tensor(opt.batchSize, nz, 1, 1)

   local label = torch.Tensor(opt.batchSize)

   local errD, errG --, errW

   local epoch_tm = torch.Timer()
   local tm = torch.Timer()
   local data_tm = torch.Timer()
----------------------------------------------------------------------------
   if opt.gpu > 0 then
      input_segment = input_segment:cuda()
      --  input_segment2 = input_segment2:cuda()

      input_image = input_image:cuda()

      noise = noise:cuda()
      label = label:cuda()
--   caffemodel:cuda()
      netD:cuda()
--   netG:cuda()

      criterion:cuda()
   end

   if opt.use_cudnn == 1 then
      cudnn = require('cudnn')
      netD = cudnn.convert(netD, cudnn)

   end

   local parametersD, gradParametersD = netD:getParameters()


   if opt.display then disp = require 'display' end

   local sample = function()
      data_tm:reset(); data_tm:resume()

      -- real_img, wrong_img, real_txt = data:getBatch()

      real_seg, real_img = data:getBatch()

      data_tm:stop()

      input_segment:copy(real_seg)
      --input_segment2:copy(wrong_seg)

      input_image:copy(real_img)
   end


--   local function weights_init(m)
--      local name = torch.type(m)
--      if name:find('Convolution') then
--         m.weight:normal(0.0, 0.01)
--         m.bias:fill(0)
--      elseif name:find('BatchNormalization') then
--         if m.weight then m.weight:normal(1.0, 0.02) end
--         if m.bias then m.bias:fill(0) end
--      end
--   end

-- create closure to evaluate f(X) and df/dX of discriminator
   fake_score = 0.5
   local fDx = function(x)
--   netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
--  netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

      gradParametersD:zero()

      -- train with real
--   label:fill(real_label)


      target =  input_segment:squeeze(2)
--   print("target size")
--   print(target:size())
--   output1 = caffemodel:forward(input_image)

      output = netD:forward( input_image)
--   print("output size")
--   print(output:size())
      errD = criterion:forward(output,target)
      df_do = criterion:backward(output, target)
--   netD:backward(input_image, df_do)
      netD:backward(input_image, criterion.gradInput)

      -- train with fake


      --errD = errD_real --+ errD_fake + errD_wrong
      -- errW = errD_wrong
      print("errD")
      print(errD)
      return errD, gradParametersD
   end

-- create closure to evaluate f(X) and df/dX of generator
--local fGx = function(x)
--netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
--  netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

-- gradParametersG:zero()

--  if opt.noise == 'uniform' then -- regenerate random noise
--    noise:uniform(-1, 1)
--  elseif opt.noise == 'normal' then
--    noise:normal(0, 1)
--  end
--  local fake = netG:forward{noise, input_image}
--  input_segment:copy(fake)
--  label:fill(real_label) -- fake labels are real for generator cost

--  local output = netD:forward{input_segment, input_image}

--  local cur_score = output:mean()
--  fake_score = 0.99 * fake_score + 0.01 * cur_score

--  errG = criterion:forward(output, label)
--  local df_do = n:backward(output, label)
--  local df_dg = netD:updateGradInput({input_segment, input_image}, df_do)

--  netG:backward({noise, input_image}, df_dg[1])
--  return errG, gradParametersG
--end

   sgd_params = {
      learningRate = 1e-10,
      learningRateDecay = 1e-4,
      weightDecay = 0.0005,
      momentum = 0.9
   }

-- train
   for epoch = 1, opt.niter do
      epoch_tm:reset()

      if epoch % opt.decay_every == 0 then
--    optimStateG.learningRate = optimStateG.learningRate * opt.lr_decay
         optimStateD.learningRate = optimStateD.learningRate * opt.lr_decay
      end

      for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
         tm:reset()

         sample()

         optim.adam(fDx, parametersD, optimStateD)
--      optim.sgd(fDx,parametersD,sgd_params)
--    optim.adam(fGx, parametersG, optimStateG)

         -- logging
         if ((i-1) / opt.batchSize) % opt.print_every == 0 then
            print(('[%d][%d/%d] T:%.3f  DT:%.3f lr: %.4g '
                  .. ' D:%.3f '):format(
                  epoch, ((i-1) / opt.batchSize),
                  math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                  tm:time().real, data_tm:time().real,
                  optimStateD.learningRate,
                  errD and errD or -1 ))


            disp.image(real_img, {win=opt.display_id * 3, title=opt.name})
         end
--    if ((i-1) / opt.batchSize) % opt.print_every == 0 then
--      print(('[%d][%d/%d] T:%.3f  DT:%.3f lr: %.4g '
--                .. ' G:%.3f  D:%.3f W:%.3f fs:%.2f'):format(
--              epoch, ((i-1) / opt.batchSize),
--              math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
--              tm:time().real, data_tm:time().real,
--              optimStateG.learningRate,
--              errG and errG or -1, errD and errD or -1,
--              errW and errW or -1, fake_score))
--      local fake = netG.output
--      disp.image(fake:narrow(1,1,opt.batchSize), {win=opt.display_id, title=opt.name})
--      disp.image(real_img, {win=opt.display_id * 3, title=opt.name})
--    end
      end
      if epoch % opt.save_every == 0 then
         paths.mkdir(opt.checkpoint_dir)
         --torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG)
         print(netD)

         torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_D.net', netD:clearState())
         torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_opt.t7', opt)
         print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
               epoch, opt.niter, epoch_tm:time().real))
      end
   end

