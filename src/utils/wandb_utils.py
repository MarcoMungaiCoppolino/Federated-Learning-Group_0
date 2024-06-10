import wandb

class WandbLogger:
    def __init__(self, args):
        self.key = args.wandb_key
        self.run = None
        if args.wandb_key:
            wandb.login(key=args.wandb_key)
            self.run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, entity=args.wandb_username)
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

    def get_execution_link(self):
        if self.run:
            return self.run.get_url()
        return "No active run"

# Example usage:
# args should be an object with the required wandb attributes
# args = Namespace(wandb_key='your_key', wandb_project='your_project', wandb_run_name='your_run_name', wandb_username='your_username')
# wandb_logger = WandbLogger(args)
# print(wandb_logger.get_execution_link())

__all__ = ['WandbLogger']
