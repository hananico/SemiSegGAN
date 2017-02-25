require 'image'

require 'xlua'
require 'optim'
require 'hdf5'
dir = require 'pl.dir'

trainLoader = {}
require 'loadcaffe'

-- this will load the network and print it's structure





local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
local dict = {}
for i = 1,#alphabet do
    dict[alphabet:sub(i,i)] = i
end
ivocab = {}
for k,v in pairs(dict) do
  ivocab[v] = k
end
alphabet_size = #alphabet

function decode(txt)
  local str = ''
  for w_ix = 1,txt:size(1) do
    local ch_ix = txt[w_ix]
    local ch = ivocab[ch_ix]
    if (ch  ~= nil) then
      str = str .. ch
    end
  end
  return str
end

function trainLoader:decode2(txt)
  local str = ''
  _, ch_ixs = txt:max(2)
  for w_ix = 1,txt:size(1) do
    local ch_ix = ch_ixs[{w_ix,1}]
    local ch = ivocab[ch_ix]
    if (ch ~= nil) then
      str = str .. ch
    end
  end
  return str
end

-- local prototxt = '/home/student/caffe_orig/caffe-master/models/VGG_ILSVRC_16_layers_deploy_conv.prototxt'
-- local binary = '/home/student/caffe_orig/caffe-master/models/VGG_ILSVRC_16_layers_conv.caffemodel'
--net = loadcaffe.load(prototxt, binary)

trainLoader.alphabet = alphabet
trainLoader.alphabet_size = alphabet_size
trainLoader.dict = dict
trainLoader.ivocab = ivocab
trainLoader.decode = decoder

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

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(path, depth, filter255)
  collectgarbage()
  -- TODO: we need to augment or pad the input image here
   local input = loadImage(path, depth, filter255)
	
	return input
	


end


local function readh5(path)
      	local myFile = hdf5.open(path, 'r')
	local data = myFile:read('feat'):all()
       -- local class = myFile:read('class'):all()
       
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


  --local data_seq1 = torch.zeros(quantity, opt.nt) -- real eeg
  --local data_seq2 = torch.zeros(quantity,  ) -- wrong image
  -- local data_img = torch.zeros(quantity, 3, sampleSize[2], sampleSize[2]) -- real image
  --local data_eeg = torch.zeros(quantity, opt.txtSize) -- real image
  
  local data_img1 = torch.Tensor(quantity, 3, sampleSize[2], sampleSize[2]) -- real image
  local data_img2 = torch.Tensor(quantity, 3, sampleSize[2], sampleSize[2]) -- wrong image
  local data_eeg1 = torch.zeros(quantity , opt.txtSize) -- real eeg
  local ids = torch.zeros(quantity)


  for n = 1, quantity do
    -- robust loading of files.
    local loaded = false
    local info1, info2, info3

		infos = explode(" ",file_samples[ix_file1[n]])
		info1 = infos[1]
		info2 = infos[2] -- Fixed wrong image
		info3 = infos[3]

		-- Randomized wrong segment
		--infos = explode(" ",file_samples[ix_file2[n]])
		--info2 = infos[2]


    local img_file1 = opt.img_dir .. '/' .. info1
     
    local img1 = trainHook(img_file1,3,0)
 
    local img_file2 = opt.img_dir .. '/' .. info2
    local img2 = trainHook(img_file2,3,0)
    local eeg_file = opt.feat_dir  .. '/' .. info3
     
    local eeg = readh5(eeg_file)
		--local data_seg1 = opt.img_dir .. '/' .. info1
    --local seg1 = trainHook(data_seg1, 1, true)
		--local data_seg2 = opt.img_dir .. '/' .. info2
    --local seg2 = trainHook(data_seg2, 1, true)
		--local filename_img = opt.img_dir .. '/' .. info3
    --local fc7_path = opt.feat_dir .. '/' 
    -- local img = trainHook(filename_img, 3)
    --local img = trainHook(filename_img, 3, false)

    --print("real img: " .. img_file1  .. ", Wrong img: " .. img_file2 .. ", Right eeg: " .. eeg_file )
    -- print("loading image" .. filename_img)

		-- print("Nasim - Segment1: " .. data_seg1)
		-- print("Nasim - Segment2: " .. data_seg2)
		-- print("Nasim - Image: " .. filename_img)
		-- print(data_seg2)
		-- print(img)
    -- real image

   
    data_img1[n]:copy(img1)
    --print('copy real image')
   
      -- mis-match image
    data_img2[n]:copy(img2)
    --print('copy wrong image')

    -- real eeg
   
    data_eeg1[n]:copy(eeg)
    --print('copy eeg')
--    data_img[n]:copy(torch.reshape(eeg, opt.txtSize))
    
    -- local fc7 = loadcaffeFC7(net, filename_img)
      -- i = string.find(info3, "/")
      -- j = string.find(info3, ".png")
      
      -- feat_path = opt.feat_dir .. '/' .. string.sub(info3, i+1, j-1) .. '.h5'
      -- local fc7 =  readh5(feat_path)
      -- data_img[n]:copy(torch.reshape(fc7, opt.txtSize):float())

  end
  
  collectgarbage(); collectgarbage()
 -- print("batch loaded")
  return data_img1, data_img2, data_eeg1
end

function trainLoader:size()
  return size
end



