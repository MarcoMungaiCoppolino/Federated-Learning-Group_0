import wandb

class WandbLogger:
    def __init__(self, args):
        self.key = args.wandb_key
        if args.wandb_key:
            wandb.login(key=args.wandb_key)
            wandb.init(project=args.wandb_project, name=args.wandb_run_name, entity=args.wandb_username)
            wandb.config.update(args)
        else:
            print("No wandb key provided")
    def watch(self, model):
        if self.key:
            wandb.watch(model)
    def log(self, log):
        if self.key:
            wandb.log(log)

    def save(self, filename):
        if self.key:
            wandb.save(filename)


__all__ = ['WandbLogger']