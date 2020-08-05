# Code Completion

## Setup

All commands have been tested on Linux. If you use Windows run the commands through `WSL` or `Git Bash`.

The following command will setup the `conda` environment.
```bash
$ source ./init.sh
```

If you also want to download the needed dataset you can pass the optional `--with-dataset` flag.
```bash
$ source ./init.sh --with-dataset
```

## Updating dependencies

1. Change the `environment.yml` file according to the [docs](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#updating-an-environment).
2. Run the following command
```bash
$ source ./update_dependencies.sh
```
