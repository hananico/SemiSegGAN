require 'image'

require 'xlua'
require 'optim'
require 'hdf5'
dir = require 'pl.dir'

trainLoader = {}
require 'loadcaffe'

-- this will load the network and print it's structure







-- local prototxt = '/home/student/caffe_orig/caffe-master/models/VGG_ILSVRC_16_layers_deploy_conv.prototxt'
-- local binary = '/home/student/caffe_orig/caffe-master/models/VGG_ILSVRC_16_layers_conv.caffemodel'
--net = loadcaffe.load(prototxt, binary)

print("Reading batch file: " .. opt.batchset)

local file_samples = {}
for line in io.lines(opt.batchset) do
  file_samples[#file_samples + 1] = line
end


local files = {}
local size = 0
-- cur_files = dir.getfiles(opt.data_root)
size = size + #file_samples

local loadSize   = {3, opt.loadSize}
local sampleSize = {3, opt.fineSize}

local function explode(div,str) -- credit: http://richard.warburton.it
  if (div=='') then return false end
  local pos,arr = 0,{}
  -- for each divider found
  for st,sp in function() return string.find(str,div,pos,true) end do
  table.insert(arr,string.sub(str,pos,st-1)) -- Attach chars left of current divider
  pos = sp + 1 -- Jump past current divider
end
table.insert(arr,string.sub(str,pos)) -- Attach chars right of last divider
return arr
end

local function loadImage(path, depth, filter255)
  local input
  if depth == 1 then
    input= image.load(path, depth, 'float')
--    print(input:type())
--    print ("input min")
--    print(input:min())
--    print ("input max")
--    print(input:())
    input = input + torch.ones(input:size())
  else
    input= image.load(path, depth, 'float')
  end


  if filter255 then
    input[input:gt(1)] = 255


--  print ("input after filter min")
--  print(input:min())
--  print ("input max")
--  print(input:max())
  end

  input = image.scale(input, loadSize[2], loadSize[2])

  return input
end

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(path, depth, filter255)
  collectgarbage()

  -- TODO: we need to augment or pad the input image here
  local input = loadImage(path, depth, filter255)
--print("trainHook" )
--  print( input:size())
  return input
  --   if opt.no_aug == 1 then
  --     return image.scale(input, sampleSize[2], sampleSize[2])
  --   end
  --
  --   local iW = input:size(3)
  --   local iH = input:size(2)
  --
  --   -- do random crop
  --   local oW = sampleSize[2];
  --   local oH = sampleSize[2]
  --   local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
  --   local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
  --

  -- print("Nasim - Image: " .. iW .. "x" .. iH .." ==>  crop: " .. w1 + oW .. "x" .. h1 + oH)
  --
  --   local out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
  --   assert(out:size(2) == oW)
  --   assert(out:size(3) == oH)
  --   -- do hflip with probability 0.5
  --   if torch.uniform() > 0.5 then out = image.hflip(out); end
  --   -- rotation
  --   if opt.rot_aug then
  --     local randRot = torch.rand(1)*0.16-0.08
  --     out = image.rotate(out, randRot:float()[1], 'bilinear')
  --   end
  --   out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]
  -- return out
end

local function readh5(path)
  local myFile = hdf5.open(path, 'r')
  local data = myFile:read('fc7'):all()
  myFile:close()
  return data
end

local function loadcaffeFC7(net, path)
  -- input = torch.randn(10,3,224,224):double()

  local input = image.load(path, 3, 'float')

  input = image.scale(input, 224, 224)
  input = torch.reshape(input, 1, 3, 224, 224):double()
  output = net:forward(input)
  -- gradInput = net:backward(input, output)
  local conv_nodes = net:findModules('nn.SpatialConvolution')
  return conv_nodes[#conv_nodes-1].output
end


function try(f, catch_f)
  local status, exception = pcall(f)
  if not status then
    catch_f(exception)
  end
end

function trainLoader:sample(quantity)

  local ix_file1 = torch.Tensor(quantity)
  local ix_file2 = torch.Tensor(quantity)

  for n = 1, quantity do
    ix_file1[n] = math.random(size)

    -- Create another random number that is not equal to the last one
    ix_file2[n] = math.random(size)
    while (ix_file2[n] == ix_file1[n]) do
      ix_file2[n] = math.random(size)
    end
  end


  local data_seq1 =  torch.FloatTensor(quantity, 1, sampleSize[2], sampleSize[2]) -- real Segment
  --local data_seq2 = torch.zeros(quantity, 1, sampleSize[2], sampleSize[2]) -- wrong Segment
  local data_img =  torch.FloatTensor(quantity, 3, sampleSize[2], sampleSize[2]) -- real image


  for n = 1, quantity do
    -- robust loading of files.
    local loaded = false
    local info1, info2, info3

  infos = explode(" ",file_samples[ix_file1[n]])

  info1 = infos[1]
  -- info2 = infos[2] -- Fixed wrong segment
  info2 = infos[2]

--		-- Randomized wrong segment
--		infos = explode(" ",file_samples[ix_file2[n]])
--		info2 = infos[2]


  local data_seg1 = opt.img_dir .. '/' .. info1
  local seg1 = trainHook(data_seg1, 1, true)
--		local data_seg2 = opt.img_dir .. '/' .. info2
--    local seg2 = trainHook(data_seg2, 1, true)
  local filename_img = opt.img_dir .. '/' .. info2
  -- local fc7_path = opt.feat_dir .. '/'
  -- local img = trainHook(filename_img, 3)
--    print(filename_img)
  local img = trainHook(filename_img, 3, false)

  data_seq1[n]:copy(seg1)

  -- real image
  --data_img[n]:copy(torch.reshape(img, opt.txtSize))
  data_img[n]:copy(img)
  -- local fc7 = loadcaffeFC7(net, filename_img)
  -- i = string.find(info3, "/")
  -- j = string.find(info3, ".png")

  -- feat_path = opt.feat_dir .. '/' .. string.sub(info3, i+1, j-1) .. '.h5'
  -- local fc7 =  readh5(feat_path)
  -- data_img[n]:copy(torch.reshape(fc7, opt.txtSize):float())

end
collectgarbage(); collectgarbage()
return data_seq1, data_img
end

function trainLoader:size()
  return size
end



