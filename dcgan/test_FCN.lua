--[[

Generic training script for GAN, GAN-CLS, GAN-INT, GAN-CLS-INT.

--]]
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
--require 'caffe'
require 'cudnn'
require 'dpnn'
require 'nnx'
require 'itorch'


local ffi = require 'ffi'
require 'loadcaffe'
local C = loadcaffe.C

opt = {
  numCaption = 1,
  large = 0,
  save_every = 100,
  print_every = 1,
  dataset = 'voc',
  img_dir = '/home/student/caffe_orig/caffe-master/data/VOC2012_SEG_AUG',
  feat_dir='/home/student/icml2016/VGG_fc7' ,
  cls_weight = 0.05,
  filenames = '',
  data_root = '/home/student/caffe_orig/caffe-master/data/VOC2012_SEG_AUG',
  checkpoint_dir = '/home/student/icml2016/checkpoints/',
  batchset = './batch_voc_train.txt' ,
  batchSize = 10,  -- 64 -> 10
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

----------------------------------------------------------------------------
local function weights_init(m)
  local name = torch.type(m)
  if name:find('Convolution') then
    m.weight:normal(0.0, 0.02)
    m.bias:fill(0)
  elseif name:find('BatchNormalization') then
    if m.weight then m.weight:normal(1.0, 0.02) end
    if m.bias then m.bias:fill(0) end
  end
end

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
--  netD:cuda()
--   netG:cuda()

  criterion:cuda()
end

if opt.use_cudnn == 1 then

  cudnn = require('cudnn')
--  netD = cudnn.convert(netD, cudnn)

end




--local parametersD, gradParametersD = netD:getParameters()


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

-- create closure to evaluate f(X) and df/dX of discriminator
fake_score = 0.5


--local fDx = function(x)
----   netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
----  netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

--  gradParametersD:zero()

--  -- train with real
----   label:fill(real_label)


--  target =  input_segment:squeeze(2)
----   print("target size")
----   print(target:size())
----   output1 = caffemodel:forward(input_image)

--  output = netD:forward( input_image)
----   print("output size")
----   print(output:size())
--  errD = criterion:forward(output,target)
--  df_do = criterion:backward(output, target)
----   netD:backward(input_image, df_do)
--  netD:backward(input_image, criterion.gradInput)

--  -- train with fake


--  --errD = errD_real --+ errD_fake + errD_wrong
--  -- errW = errD_wrong
--  print("errD")
--  print(errD)
--  return errD, gradParametersD
--end


sgd_params = {
  learningRate = 1e-10,
  learningRateDecay = 1e-4,
  weightDecay = 0.0005,
  momentum = 0.9
}

-------Test---------------------------------------------
for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
  tm:reset()
  print("test")
  sample()

  target =  input_segment:squeeze(2)
--   print("target size")
--   print(target:size())
--   output1 = caffemodel:forward(input_image)

  netD = torch.load('/home/student/icml2016/checkpoints/semiGAN_300_net_D.t7' )

  print(netD)
  output = netD:forward( input_image)
  print(output:size())
  --model.output:size()

  idxs, maxs = torch.max(output, 2)

  print(idxs:size())
  print(maxs:size())
  print(idxs[2]:squeeze(1))

  itorch.image(maxs[1]:squeeze(1))
  itorch.image(trainer.target)
end






