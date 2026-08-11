import wandb
api = wandb.Api()
projects = api.projects(entity="awesome-prompt")
for project in projects:
    print(project.name)
