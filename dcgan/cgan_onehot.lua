
-- Options table
opt = {}
help = {}
-- Checkpoint options
opt.checkpoint_dir = "checkpoints/"; help.checkpoint_dir = "Checkpoint directory (must exist)"
opt.tag = ""; help.tag = "Checkpoint tag (added to file name)"
opt.resume = ""; help.resume = "Load checkpoint"
-- Dataset options
opt.dataset = "cifar10_images"; help.dataset = "Dataset directory"
opt.zero_cond = false; help.zero_cond = ""
opt.num_threads = 8
-- Model options
opt.noise = 100; help.noise = "Noise vector size"
opt.g_model = "code"; help.g_model = "Generator model name ('file' to load from file)"
opt.g_model_path = ""; help.g_model_path = "Generator model path (for model 'file')"
opt.g_filters = 64; help.g_filters = "Number of filters in generator's last layer"
opt.d_model = "code"; help.d_model = "Discriminator model name ('file' to load from file)"
opt.d_model_path = ""; help.d_model_path = "Discriminator model path (for model 'file')"
opt.d_filters = 64; help.d_filters = "Number of filters in discriminator's first layer"
-- Training options
opt.seed = 1; help.seed = "Random seed"
opt.batch_size = 16; help.batch_size = "Batch size"
opt.g_optim = "adam"; help.optim = "Optimization algorithm (generator; e.g. sgd, adam, rmsprop --- basically any supported by optim)"
opt.d_optim = "adam"; help.optim = "Optimization algorithm (discriminator; e.g. sgd, adam, rmsprop --- basically any supported by optim)"
opt.g_lr = -1; help.g_lr = "Learning rate (generator)"
opt.d_lr = -1; help.d_lr = "Learning rate (discriminator)"
opt.lrd = 0.00001; help.lrd = "Learning rate decay"
opt.wd = 0.0005; help.wd = "Weight decay"
opt.mom = 0.9; help.mom = "Weight momentum"
opt.epochs = 100; help.epoch = "Number of epochs"
opt.save_every = 10; help.save_every = "Save checkpoint every X epochs"
opt.profile = false; help.profile = "Profile times"
opt.display = 10; help.display = "Display details for N images per epoch (itorch only)"
-- Backend options
opt.cuda = false; help.cuda = "Use CUDA"
opt.cudnn = false; help.cudnn = "Use CUDNN"

-- Overwrite options in itorch
if itorch then
    opt.cuda = true
    opt.cudnn = true
    opt.tag = "cgan-zerocond"
    opt.zero_cond = true
    opt.d_optim = "sgd"
    opt.g_optim = "sgd"
    opt.d_lr = 0.004
    opt.g_lr = 0.01
    --opt.save_every = 20
    --opt.resume = 'checkpoints/checkpoint-000100.t7'
end

-- Read from command line
if not itorch then
    -- Command-line options
    cmd = torch.CmdLine()
    -- Add options
    for k,v in pairs(opt) do
        cmd:option("-" .. k, v, help[k])
    end
    -- Read options
    opt = cmd:parse(arg)
end
-- Parse seed
if opt.seed == -1 then
    opt.seed = 1
end
-- If negative learning rate, unset it (for using default values in certain algorithms, e.g. Adam)
if opt.g_lr < 0 then
    opt.g_lr = nil
end
if opt.d_lr < 0 then
    opt.d_lr = nil
end
-- Set global CUDA flag
cuda = opt.cuda
print(opt)

-- Setup requires
package.path = package.path .. ";loaders/?.lua;utils/?.lua;models/?.lua"

-- Requires
require "nn"
require "optim"
if cuda then
    print("Importing CUDA libs")
    require "cutorch"
    require "cunn"
    if opt.cudnn then
        require "cudnn"
    end
end
torch_utils = require "torch_utils"
if cuda then
    print_cuda_free()
end

-- Check checkpoints exist for this experiment
if opt.resume == "" and opt.tag ~= "" and paths.filep(opt.checkpoint_dir .. "/history-" .. opt.tag .. ".t7") then
    print("ERROR: found history file for this experiment tag. Remove it and all related checkpoints")
    if not itorch then
        os.exit(1)
    end
end

-- Options for number visualization
time_format = "%.1f"
loss_format = "%.5f"

-- Setup Torch
random_seed = opt.seed
if random_seed then
    torch.manualSeed(random_seed)
end
torch.setdefaulttensortype("torch.FloatTensor")

-- Initialize loader
package.loaded["DirLoader"] = nil
loader_driver = require "DirLoader"
loader = loader_driver.load({
        paths = {opt.dataset},
        splits = {train = 1},
        thread_code = "cifar10_thread_code.lua",
        num_threads = opt.num_threads,
        batch_size = opt.batch_size,
        crop_size = 27
})
-- Compute number of batches
splits = loader:get_split_names()
num_batches = math.floor(loader.splits.train.size/opt.batch_size)
-- Initialize current batches (to keep track of epoch progress)
curr_batch = 0
-- Start loading
loader:start_loading("train")

-- One-hot encoding
function onehot(dst, labels)
    local batch_size = labels:nElement()
    dst:zero()
    for i=1,batch_size do
        dst[{i,labels[i]}] = 1
    end
end

-- Weight initialization function
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

-- Model options
nc = 3
nz = opt.noise
ncnd = #loader.dataset.classes
ndf = opt.d_filters
ngf = opt.g_filters
real_label = 1
fake_label = 0

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

-- Generator
if opt.g_model == "file" then
    -- Load from file
    g_net = torch.load(opt.g_model_path)
elseif opt.g_model == "code" then
    -- Create from code
    g_net = nn.Sequential()
    local g_par = nn.ParallelTable()
    g_par:add(nn.Identity())
    g_par:add(nn.View(ncnd,1,1):setNumInputDims(1))
    g_net:add(g_par)
    g_net:add(nn.JoinTable(1,3))
    -- input is noise+condition, going into a convolution
    g_net:add(SpatialFullConvolution(nz + ncnd, ngf * 8, 3, 3))
    g_net:add(SpatialBatchNormalization(ngf * 8))
    g_net:add(nn.ReLU(true))
    -- state size: (ngf*8) x 3 x 3
    g_net:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
    g_net:add(SpatialBatchNormalization(ngf * 4))
    g_net:add(nn.ReLU(true))
    -- state size: (ngf*4) x 6 x 6
    g_net:add(SpatialFullConvolution(ngf * 4, ngf * 3, 4, 4, 2, 2, 1, 1))
    g_net:add(SpatialBatchNormalization(ngf * 3))
    g_net:add(nn.ReLU(true))
    -- state size: (ngf*3) x 12 x 12
    g_net:add(SpatialFullConvolution(ngf * 3, ngf*2, 4, 4, 2, 2, 1, 1))
    g_net:add(SpatialBatchNormalization(ngf*2))
    g_net:add(nn.ReLU(true))
    -- state size: (ngf*2) x 24 x 24
    g_net:add(SpatialFullConvolution(ngf*2, nc, 4, 4))
    g_net:add(nn.Tanh())
    -- state size: (nc) x 27 x 27
else
    -- Create from driver
    local model_driver = require(opt.g_model)
    g_net = model_driver.new({})
end

-- Discriminator
if opt.d_model == "file" then
    -- Load from file
    d_net = torch.load(opt.d_model_path)
elseif opt.d_model == "code" then
    -- Create from code
    d_net = nn.Sequential()

    -- Create convolutional path
    d_conv_net = nn.Sequential()
    -- input is (nc) x 27 x 27
    d_conv_net:add(SpatialConvolution(nc, ndf, 4, 4))
    --d_net:add(SpatialBatchNormalization(ndf))
    d_conv_net:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf) x 24 x 24
    d_conv_net:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
    --d_net:add(SpatialBatchNormalization(ndf * 2))
    d_conv_net:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*2) x 12 x 12
    d_conv_net:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
    --d_conv_net:add(SpatialBatchNormalization(ndf * 4))
    d_conv_net:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*4) x 6 x 6
    d_conv_net:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
    d_conv_net:add(SpatialBatchNormalization(ndf * 8))
    d_conv_net:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*8) x 3 x 3

    --d_conv_net:add(SpatialConvolution(ndf * 8, 512, 3, 3))
    --d_conv_net:add(SpatialBatchNormalization(512))
    --d_conv_net:add(nn.LeakyReLU(0.2, true))
    -- state size: 512 x 1 x 1
    --d_conv_net:add(SpatialConvolution(512, 64, 1, 1))
    --d_conv_net:add(SpatialBatchNormalization(64))
    --d_conv_net:add(nn.LeakyReLU(0.2, true))
    -- state size: 64 x 1 x 1
    --d_conv_net:add(nn.View(-1):setNumInputDims(3))
    -- state size: 64

    -- Create condition path
    d_cond_net = nn.Sequential()
    d_cond_net:add(nn.Replicate(3,3))
    d_cond_net:add(nn.Replicate(3,4))

    -- Create parallel conv/condition path
    d_par = nn.ParallelTable()
    d_par:add(d_conv_net)
    d_par:add(d_cond_net)

    -- Add parallel path to discriminator
    d_net:add(d_par)

    -- Join convolution and condition
    d_join = nn.JoinTable(1,3)
    d_net:add(d_join)
    -- state size: (ndf*8 + ncnd) x 3 x 3
    d_net:add(SpatialConvolution(ndf*8 + ncnd, ndf*16, 3, 3))
    d_net:add(SpatialBatchNormalization(ndf*16))
    d_net:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*16) x 1 x 1
    d_net:add(SpatialConvolution(ndf*16, 1, 1, 1))
    d_net:add(nn.Sigmoid())
    -- state size: 1 x 1 x 1
else
    -- Create from driver
    local model_driver = require(opt.model)
    d_net = model_driver.new({})
end
-- Initialize weights
d_net:apply(weights_init)

-- Test network output
local g_test_input = {torch.Tensor(1, nz, 1, 1), torch.Tensor(1, ncnd)}
local g_output_size = g_net:forward(g_test_input):size()
print("Generator output size:")
print(g_output_size)
local d_test_input = {torch.Tensor(1, 3, 27, 27), torch.Tensor(1, ncnd)}
local d_output_size = d_net:forward(d_test_input):size()
print("Discriminator output size:")
print(d_output_size)

-- Training options
g_training_method = opt.g_optim
d_training_method = opt.d_optim
g_train_params = {learningRate = opt.g_lr, learningRateDecay = opt.lrd, weightDecay = opt.wd, momentum = opt.mom, beta1 = opt.mom, epsilon = opt.eps}
d_train_params = {learningRate = opt.d_lr, learningRateDecay = opt.lrd, weightDecay = opt.wd, momentum = opt.mom, beta1 = opt.mom, epsilon = opt.eps}
criterion = nn.BCECriterion()
g_current_train_params = nil
d_current_train_params = nil
batch = torch.Tensor()
cond = torch.Tensor(opt.batch_size, ncnd)
noise = torch.Tensor(opt.batch_size, nz, 1, 1)
labels = torch.Tensor(opt.batch_size)

-- Start epoch
start_epoch = 0

-- History
history = {e = {}, g_loss = {}, d_real_right_loss = {}, d_real_wrong_loss = {}, d_fake_loss = {}}

-- Load checkpoint
if opt.resume ~= "" then
    print("Loading saved model")
    local loaded_data = torch.load(opt.resume)
    -- Overwrite data
    g_net = loaded_data.g_net
    d_net = loaded_data.d_net
    start_epoch = loaded_data.epoch + 1
    g_train_params = loaded_data.g_train_params
    d_train_params = loaded_data.d_train_params
    if cuda then
        for param_name,param_value in pairs(d_train_params) do
            if type(param_value) == "userdata" and param_value:type() == "torch.FloatTensor" then
                d_train_params[param_name] = param_value:cuda()
            end
        end
        for param_name,param_value in pairs(g_train_params) do
            if type(param_value) == "userdata" and param_value:type() == "torch.FloatTensor" then
                g_train_params[param_name] = param_value:cuda()
            end
        end
    end
    history = loaded_data.history
    -- Convert history data back to tables
    for key,value in pairs(history) do
        if type(value) == "userdata" and value:type() == "torch.FloatTensor" then
            history[key] = torch.totable(value)
        end
    end
end

-- Copy model to CUDA
if cuda then
    print("Copying model to CUDA")
    -- Check CUDNN
    if opt.cudnn then
        g_net = cudnn.convert(g_net, cudnn)
        d_net = cudnn.convert(d_net, cudnn)
    end
    batch = torch.CudaTensor()
    cond = cond:cuda()
    noise = noise:cuda()
    labels = labels:cuda()
    g_net:cuda()
    d_net:cuda()
    criterion:cuda()
    print_cuda_free()
end

-- Describe configuration
print("Training method (generator): " .. g_training_method)
print("Training parameters (generator): ")
print(g_train_params)
print("Training method (discriminator): " .. d_training_method)
print("Training parameters (discriminator): ")
print(d_train_params)

-- Get parameters and derivatives of loss
g_w, g_dl_dw = g_net:getParameters()
d_w, d_dl_dw = d_net:getParameters()

-- Performance monitoring
print_info_cnt = 0
print_info_every = 2
show_samples_cnt = 0
show_samples_per_epoch = 3
show_samples_every = math.floor(num_batches/show_samples_per_epoch)

-- Data to be shown (set during d_feval and g_feval)
show_real_batch = torch.Tensor()
show_fake_batch = torch.Tensor()
show_real_right_loss = 0
show_real_wrong_loss = 0
show_fake_loss = 0
show_g_loss = 0

-- Check if show-data should be updated
function show_this_batch()
    return show_samples_cnt % show_samples_every == 0
end

if itorch then
    Plot = require "itorch.Plot"
end

function d_show_training_info()
    -- Update counts
    print_info_cnt = print_info_cnt + 1
    show_samples_cnt = show_samples_cnt + 1
    check_interrupt_cnt = check_interrupt_cnt + 1
    -- Check frontend
    if itorch then
        -- Show samples
        d_show_samples()
    -- Epoch info is shown by g_show_training_info
    end
end

-- Show samples during training (discriminator)
function d_show_samples()
    if show_this_batch() then
        -- Show real batch
        itorch.image(show_real_batch)
        -- Show fake batch
        itorch.image(show_fake_batch)
        -- Show losses
        print("Discriminator loss: real/right: " .. show_real_right_loss .. ", real/wrong: " .. show_real_wrong_loss .. ", fake: " .. show_fake_loss ..")")
    end
end

function g_show_training_info()
    -- Check frontend
    if itorch then
        -- Show samples
        g_show_samples()
    else
        -- Print epoch info
        print_epoch_info()
    end
end

-- Show loss during training (generator)
function g_show_samples()
    if show_samples_cnt % show_samples_every == 0 then
        -- Show losses
        print("Generator loss: " .. show_g_loss)
        -- Print epoch info
        print_epoch_info(true)
    end
end

-- Print epoch info
function print_epoch_info(force)
    -- Doesn't work well with itorch
    if itorch and not force then
        return
    end
    if force or print_info_cnt % print_info_every == 0 then
        -- Print summary
        io.write("\r")
        io.write("Epoch " .. epoch)
        io.write(": DL=" .. string.format(loss_format, (d_real_right_loss_acc + d_real_wrong_loss_acc + d_fake_loss_acc)/(d_real_right_loss_cnt + d_real_wrong_loss_cnt + d_fake_loss_cnt)))
        io.write(" (real/right=" .. string.format(loss_format, d_real_right_loss_acc/d_real_right_loss_cnt))
        io.write(", real/wrong=" .. string.format(loss_format, d_real_wrong_loss_acc/d_real_wrong_loss_cnt))
        io.write(", fake=" .. string.format(loss_format, d_fake_loss_acc/d_fake_loss_cnt))
        io.write("), GL=" .. string.format(loss_format, g_loss_acc/g_loss_cnt))
        if progress < 1 then
            io.write(" (" .. string.format("%.0f", progress*100) .. "%)")
        end
        io.write(" (ET: " .. string.format(time_format, epoch_timer:time().real))
        if opt.profile then io.write(
            ", TBPT: " .. string.format(time_format, train_batch_preparation_time) ..
            ", TFT: " .. string.format(time_format, train_forward_time) ..
            ", TBT: " .. string.format(time_format, train_backward_time)
        ) end
        io.write(")                 ")
        if force then
            print("")
        end
    end
end

-- Plot loss and accuracy
function plot_loss()
    -- FIXME
    if false and itorch then
        -- Initialize plots
        plot = Plot():title("Loss"):legend(true)
        -- Keys and colors
        loss_keys = {"d_real_right_loss", "d_real_wrong_loss", "d_fake_loss", "g_loss"}
        loss_labels = {d_real_right_loss = "DRRL", d_real_wrong_loss = "DRWL", d_fake_loss = "DFL", g_loss = "GL"}
        loss_colors = {d_real_right_loss = "green", d_real_wrong_loss = "yellow", d_fake_loss = "blue", g_loss = "red"}
        -- Process each loss
        for _,loss in pairs(loss_keys) do
            -- Plot time trend
            plot:line(history[loss].e, history[loss], loss_colors[loss], loss_labels[loss])
        end
        -- Draw plot
        plot:draw()
    end
end

function d_feval(w)
    -- Start timer
    local aux_timer
    if opt.profile then aux_timer = torch.Timer() end
    -- Zero gradients
    d_dl_dw:zero()
    -- Increase batch count
    curr_batch = curr_batch + 1

    -- Get samples (real image, right condition)
    local batch_tmp,cond_tmp = loader:get_batch()
    batch:resize(batch_tmp:size()):copy(batch_tmp)
    if opt.zero_cond then
        cond:zero()
    else
        onehot(cond, cond_tmp)
    end
    labels:fill(real_label)
    if show_this_batch() then show_real_batch:resize(batch_tmp:size()); show_real_batch:copy(batch_tmp) end
    if opt.profile then cuda_sync(); train_batch_preparation_time = train_batch_preparation_time + aux_timer:time().real end
    -- Forward pass (real image, right condition)
    if opt.profile then aux_timer:stop(); aux_timer:reset(); aux_timer:resume() end
    local net_out = d_net:forward({batch,cond})
    -- Compute loss (real image, right condition)
    local real_right_net_loss = criterion:forward(net_out, labels)
    show_real_right_loss = real_right_net_loss -- Just set it without checking
    if opt.profile then cuda_sync(); train_forward_time = train_forward_time + aux_timer:time().real end
    -- Accumulate loss (real image, right condition)
    d_real_right_loss_acc = d_real_right_loss_acc + real_right_net_loss
    d_real_right_loss_cnt = d_real_right_loss_cnt + 1
    -- Backward pass (real image, right condition)
    if opt.profile then aux_timer:stop(); aux_timer:reset(); aux_timer:resume() end
    local dl_dy = criterion:backward(net_out, labels)
    d_net:backward({batch,cond}, dl_dy)
    if opt.profile then cuda_sync(); train_backward_time = train_backward_time + aux_timer:time().real end

    -- Get samples (real image, wrong condition)
    -- Initialize stuff, in case we don't run this part due to zero_cond being true
    local real_wrong_net_loss = 0
    if not opt.zero_cond then
        local wrong_cond = cuda and cond:float() or cond:clone()
        for i=1,opt.batch_size do
            -- Get condition for this batch element
            local cond_i = wrong_cond[i]
            -- Compute hot index
            local cond_hot_idx = cond_i:nonzero():squeeze()
            local cond_new_hot = cond_hot_idx
            while cond_new_hot == cond_hot_idx do
                cond_new_hot = torch.random(1, #loader.dataset.classes)
            end
            -- Update new hot index
            cond_i:zero()
            cond_i[cond_new_hot] = 1
        end
        if cuda then wrong_cond = wrong_cond:cuda() end
        labels:fill(fake_label)
        if opt.profile then cuda_sync(); train_batch_preparation_time = train_batch_preparation_time + aux_timer:time().real end
        -- Forward pass (real image, wrong condition)
        if opt.profile then aux_timer:stop(); aux_timer:reset(); aux_timer:resume() end
        local net_out = d_net:forward({batch,wrong_cond})
        -- Compute loss (real image, wrong condition)
        real_wrong_net_loss = criterion:forward(net_out, labels)
        show_real_wrong_loss = real_wrong_net_loss -- Just set it without checking
        if opt.profile then cuda_sync(); train_forward_time = train_forward_time + aux_timer:time().real end
        -- Accumulate loss (real image, wrong condition)
        d_real_wrong_loss_acc = d_real_wrong_loss_acc + real_wrong_net_loss
        d_real_wrong_loss_cnt = d_real_wrong_loss_cnt + 1
        -- Backward pass (real image, wrong condition)
        if opt.profile then aux_timer:stop(); aux_timer:reset(); aux_timer:resume() end
        local dl_dy = criterion:backward(net_out, labels)
        d_net:backward({batch,wrong_cond}, dl_dy)
        if opt.profile then cuda_sync(); train_backward_time = train_backward_time + aux_timer:time().real end
    end

    -- Get samples (fake; this is also forward pass for generator)
    if opt.profile then aux_timer:stop(); aux_timer:reset(); aux_timer:resume() end
    noise:normal(0,1)
    -- Generate fake batch
    local fake_batch = g_net:forward({noise,cond}) -- This is also g_net.output
    if show_this_batch() then show_fake_batch:resize(g_net.output:size()); show_fake_batch:copy(fake_batch) end
    labels:fill(fake_label)
    if opt.profile then cuda_sync(); train_forward_time = train_forward_time + aux_timer:time().real end
    -- Forward pass (fake)
    if opt.profile then aux_timer:stop(); aux_timer:reset(); aux_timer:resume() end
    local net_out = d_net:forward({fake_batch,cond}):squeeze()
    -- Compute loss (fake)
    local fake_net_loss = criterion:forward(net_out, labels)
    show_fake_loss = fake_net_loss -- Just set it without checking
    if opt.profile then cuda_sync(); train_forward_time = train_forward_time + aux_timer:time().real end
    -- Accumulate loss (fake)
    d_fake_loss_acc = d_fake_loss_acc + fake_net_loss
    d_fake_loss_cnt = d_fake_loss_cnt + 1
    -- Backward pass (fake)
    if opt.profile then aux_timer:stop(); aux_timer:reset(); aux_timer:resume() end
    local dl_dy = criterion:backward(net_out, labels)
    d_net:backward({fake_batch,cond}, dl_dy)
    if opt.profile then cuda_sync(); train_backward_time = train_backward_time + aux_timer:time().real end

    -- Update progress
    progress = curr_batch/num_batches
    -- Show samples and loss
    d_show_training_info()
    -- Check update stats
    return real_right_net_loss + real_wrong_net_loss + fake_net_loss, d_dl_dw
end

function g_feval(w)
    -- Start timer
    local aux_timer
    if opt.profile then aux_timer = torch.Timer() end
    -- Zero gradients
    g_dl_dw:zero()
    -- Forward pass (already done in d_feval)
    local net_out = d_net.output
    labels:fill(real_label)
    -- Compute loss
    local g_net_loss = criterion:forward(net_out, labels)
    show_g_loss = g_net_loss -- Just set it without checking
    if opt.profile then cuda_sync(); train_forward_time = train_forward_time + aux_timer:time().real end
    -- Accumulate loss (real)
    g_loss_acc = g_loss_acc + g_net_loss
    g_loss_cnt = g_loss_cnt + 1
    -- Backward pass
    if opt.profile then aux_timer:stop(); aux_timer:reset(); aux_timer:resume() end
    local dl_dy = criterion:backward(net_out, labels)
    local dl_dg = d_net:updateGradInput({g_net.output,cond}, dl_dy) -- cond was initially set in d_feval and not modified
    dl_dg = dl_dg[1]
    g_net:backward({noise,cond}, dl_dg)
    if opt.profile then cuda_sync(); train_backward_time = train_backward_time + aux_timer:time().real end
    -- Print partial info
    g_show_training_info()
    -- Check update stats
    return g_net_loss, g_dl_dw
end

-- Perform training for several epochs
print("Start training")
-- Set end epochs
local end_epoch = opt.epochs
if itorch then end_epoch = start_epoch+100 end
-- Check interrupt variables (training only)
check_interrupt_cnt = 0
check_interrupt_every = 5
-- Initialize loss accumulator and monitoring variables
d_real_right_loss_acc = 0
d_real_right_loss_cnt = 0
d_real_wrong_loss_acc = 0
d_real_wrong_loss_cnt = 0
d_fake_loss_acc = 0
d_fake_loss_cnt = 0
g_loss_acc = 0
g_loss_cnt = 0
progress = 0
-- Epoch loop
for e = start_epoch+1,end_epoch do
    -- Set global epoch
    epoch = e
    -- Initialize timer (we'll time epoch time regardless of opt.profile)
    epoch_timer = torch.Timer()
    -- Reset profiling variables
    train_batch_preparation_time = 0
    train_forward_time = 0
    train_backward_time = 0
    -- Reset loss accumulator and monitoring variables
    d_real_right_loss_acc = 0
    d_real_right_loss_cnt = 0
    d_real_wrong_loss_acc = 0
    d_real_wrong_loss_cnt = 0
    d_fake_loss_acc = 0
    d_fake_loss_cnt = 0
    g_loss_acc = 0
    g_loss_cnt = 0
    progress = 0
    curr_batch = 0
    -- Set training mode
    g_net:training()
    d_net:training()
    -- Go through all training batches
    for i=1,num_batches do
        -- Check interrupt
        check_interrupt_cnt = check_interrupt_cnt + 1
        if check_interrupt_cnt % check_interrupt_every == 0 and check_interrupt() then
            print("kill.txt file found, stopping")
            if itorch then
                goto stop_training
            else
                os.exit(0)
            end
        end
        -- Setup training params
        if g_current_train_params == nil then
            g_current_train_params = clone_table(g_train_params)
        end
        if d_current_train_params == nil then
            d_current_train_params = clone_table(d_train_params)
        end
        -- Perform optimization
        optim[d_training_method](d_feval, d_w, d_current_train_params)
        optim[g_training_method](g_feval, g_w, g_current_train_params)
    end
    -- Add to history
    table.insert(history.e, e)
    table.insert(history.d_real_right_loss, d_real_right_loss_acc/d_real_right_loss_cnt)
    table.insert(history.d_real_wrong_loss, d_real_wrong_loss_acc/d_real_wrong_loss_cnt)
    table.insert(history.d_fake_loss, d_fake_loss_acc/d_fake_loss_cnt)
    table.insert(history.g_loss, g_loss_acc/g_loss_cnt)
    -- Collect garbage
    collectgarbage()
    -- Compute epoch time
    cuda_sync()
    local epoch_time = epoch_timer:time().real
    -- Show final epoch info
    print_epoch_info(true)
    -- Show loss plot
    if itorch then
        plot_loss()
    end
    -- Print CUDA usage at the end of first epoch
    if e == 1 and cuda then
        print_cuda_free()
    end
    -- Generate samples
    local gen_classes = #loader.dataset.classes
    local gen_samples_per_class = 20
    local num_gen_samples = gen_classes*gen_samples_per_class
    local gen_noise = torch.Tensor(num_gen_samples, nz, 1, 1)
    gen_noise:normal(0,1)
    local gen_conds = torch.zeros(num_gen_samples, ncnd)
    for i=1,num_gen_samples do
        local gen_class = ((i-1) % gen_classes) + 1
        gen_conds[i][gen_class] = 1
    end
    if cuda then
        gen_noise = gen_noise:cuda()
        gen_conds = gen_conds:cuda()
    end
    local gen_images = g_net:forward({gen_noise,gen_conds})
    gen_images:add(1):mul(0.5)
    image.save("generated-" .. e .. ".png", image.toDisplayTensor({input = gen_images, nrow = gen_classes, padding = 1}))
    -- We save a checkpoint every now and then
    if e % opt.save_every == 0 then
        -- Clear network state
        g_net:clearState()
        d_net:clearState()
        -- Compute checkpoint and history paths
        local checkpoint_path = opt.checkpoint_dir .. "/checkpoint-";
        local history_path = opt.checkpoint_dir .. "/history-";
        if opt.tag ~= "" then
            checkpoint_path = checkpoint_path .. opt.tag .. "-"
            history_path = history_path .. opt.tag
        end
        checkpoint_path = checkpoint_path .. string.format("%06d", e) .. ".t7"
        history_path = history_path .. ".t7"
        -- Convert history data to tensor
        history_save = {e = torch.Tensor(history.e), d_real_right_loss = torch.Tensor(history.d_real_right_loss), d_real_wrong_loss = torch.Tensor(history.d_real_wrong_loss), d_fake_loss = torch.Tensor(history.d_fake_loss), g_loss = torch.Tensor(history.g_loss)}
        -- Clone train params and convert CudaTensors
        d_train_params_save = {}
        for key, value in pairs(d_current_train_params) do
            if type(value) == "userdata" and value:type() == "torch.CudaTensor" then
                d_train_params_save[key] = value:float()
            else
                d_train_params_save[key] = value
            end
        end
        g_train_params_save = {}
        for key, value in pairs(g_current_train_params) do
            if type(value) == "userdata" and value:type() == "torch.CudaTensor" then
                g_train_params_save[key] = value:float()
            else
                g_train_params_save[key] = value
            end
        end
        -- Save checkpoint and history
        print("Saving checkpoint and history")
        torch.save(checkpoint_path, {g_net = g_net:clone():float(), d_net = d_net:clone():float(), epoch = e, g_train_params = g_train_params_save, d_train_params = d_train_params_save, history = history_save})
        torch.save(history_path, history_save)
    end
end
::stop_training::
-- Overwrite start epoch for itorch
start_epoch = epoch


