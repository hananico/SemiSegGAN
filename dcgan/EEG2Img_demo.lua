
require 'image'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'lfs'
require 'hdf5'
require 'xlua'
require 'optim'

torch.setdefaulttensortype('torch.FloatTensor')

local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
local dict = {}
for i = 1,#alphabet do
    dict[alphabet:sub(i,i)] = i
end
ivocab = {}
for k,v in pairs(dict) do
  ivocab[v] = k
end

opt = {
  filenames = '',
  dataset = 'voc12',
  batchSize = 2,        -- number of samples to produce
  noisetype = 'normal',  -- type of noise distribution (uniform / normal).
  imsize = 1,            -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px, ...
  noisemode = 'random',  -- random / line / linefull1d / linefull
  gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
  display = 0,           -- Display image: 0 = false, 1 = true
  nz = 100,              
  doc_length = 201,
  queries = 'cub_queries.txt',
	base_queries_dir = '',
        feat_dir='',
  checkpoint_dir = '',
  net_gen = '',
  net_txt = '',
	loadSize = 64,
	txtSize = 12288,
}

local function loadImage(path, depth, loadSize)
  local input = image.load(path, depth, 'float')
  input = image.scale(input, loadSize, loadSize)
  return input
end

local function readh5(path)
      	local myFile = hdf5.open(path, 'r')
	local data = myFile:read('fc7'):all()
	myFile:close()
      return data
end

function saveh5(path , data)      
        print (path)
	local myFile = hdf5.open(path, 'w')	 
	myFile:write(path, data)
	myFile:close()
end

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

noise = torch.Tensor(opt.batchSize, opt.nz, opt.imsize, opt.imsize)
net_gen = torch.load(opt.checkpoint_dir .. '/' .. opt.net_gen)

net_gen:evaluate()

-- -- Extract all text features.
-- local fea_txt = {}
-- -- Decode text for sanity check.
-- local raw_txt = {}
-- local raw_img = {}
-- for query_str in io.lines(opt.queries) do
--   local txt = torch.zeros(1,opt.doc_length,#alphabet)
--   for t = 1,opt.doc_length do
--     local ch = query_str:sub(t,t)
--     local ix = dict[ch]
--     if ix ~= 0 and ix ~= nil then
--       txt[{1,t,ix}] = 1
--     end
--   end
--   raw_txt[#raw_txt+1] = query_str
--   txt = txt:cuda()
--   fea_txt[#fea_txt+1] = net_txt:forward(txt):clone()
-- end

if opt.gpu > 0 then
  require 'cunn'
  require 'cudnn'
  net_gen:cuda()
  noise = noise:cuda()
end

local html = '<html><body><h1>Generated Images</h1><table border="1px solid gray" style="width=100%"><tr><td><b>Image</b></td><td><b>Segmentation</b></td><td><b>Ground Truth</b></td></tr>'

local i = 0
for image_query in io.lines(opt.queries) do
	i = i + 1
	local image_filename = opt.base_queries_dir .. '/images/' .. image_query
         
  print('generating #' .. i .. ", file: ".. image_filename)
	local input = loadImage(image_filename, 3, opt.loadSize)
        -- j = string.find(image_query, ".png")
        ---- print(string.sub(image_query, 1, j-1))      
        -- feat_path = opt.feat_dir .. '/' .. string.sub(image_query, 1, j-1) .. '.h5'
        -- local input =  readh5(feat_path)
         
	local input_reshaped = torch.reshape(input, opt.txtSize)
  
	
  local cur_fea_txt = torch.repeatTensor(input_reshaped, opt.batchSize, 1)
  if opt.noisetype == 'uniform' then
    noise:uniform(-1, 1)
  elseif opt.noisetype == 'normal' then
    noise:normal(0, 1)
  end
  local images = net_gen:forward{noise, cur_fea_txt:cuda()}
  
  local visdir = string.format('results/%s', opt.dataset)
  lfs.mkdir('results')
  lfs.mkdir(visdir)
  
  local fname = string.format('%s/img_%d', visdir, i)
  local fname_png = fname .. '.png'
  --  saveh5(fname .. '.h5' ,  image.toDisplayTensor(images,4,opt.batchSize/2))
  images:add(1):mul(0.5)
  --image.save(fname_png, image.toDisplayTensor(images,4,torch.floor(opt.batchSize/4)))
  image.save(fname_png, image.toDisplayTensor(images,4,opt.batchSize/2))
  html = html .. string.format('\n<tr><td><img src="%s"></td><td><img width=64 height=64 src="%s"><img src="%s"><img width=64 height=64 src="%s"></td><td><img src="%s"></td></tr>',
                               "images/" .. image_query, 
																"images/" .. image_query, 
																fname_png, 
																"segmentations/" .. image_query,
																"segmentations/" .. image_query)
end

html = html .. '</html>'
fname_html = string.format('%s.html', 'res')
-- os.execute(string.format('echo "%s" > %s', html, fname_html))

file = io.open(fname_html, "w")
if file then print("File was opened: " .. fname_html) else print("What File?") end
io.output(file)
io.write(html)
io.close(file)


