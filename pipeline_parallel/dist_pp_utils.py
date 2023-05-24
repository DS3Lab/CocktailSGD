from .dist_gpipe_pipeline_async import GpipeAsync
from .dist_gpipe_no_pipeline_async import NoGpipeAsync

def get_pp_module(args, config, device, use_dp):
    
    if args.pp_mode == 'gpipe':
        return GpipeAsync(args, config, device, use_dp)
    elif args.pp_mode == 'nogpipe':
        return NoGpipeAsync(args, config, device, use_dp)
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False
        
