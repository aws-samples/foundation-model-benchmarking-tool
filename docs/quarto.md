# Setting up Quarto

`FMBench` uses [`Quarto`](https://quarto.org/) for generating reports. At the end of a run it downloads a Quarto container from `registry.gitlab.com/quarto-forge/docker/quarto quarto` and converts a markdown report into an HTML report. If however, download a Docker container is blocked in your environment you can install Quarto using the following steps. These are also described on the Quarto [website](https://quarto.org/docs/download/tarball.html).

Here are the steps, copy paste them on a Linux based Amazon EC2 instance. This is required to be done one time only. 

```{.bash}
# download Quarto tarball
wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.6.39/quarto-1.6.39-linux-amd64.tar.gz

# replace the workspace path as appropriate for your environment
WORKSPACE_PATH=~

mkdir $WORKSPACE_PATH/opt
tar -C $WORKSPACE_PATH/opt -xvzf quarto-1.6.39-linux-amd64.tar.gz
mkdir -p $WORKSPACE_PATH/.local/bin
ln -s $WORKSPACE_PATH/opt/quarto-1.6.39/bin/quarto $WORKSPACE_PATH/.local/bin/quarto
( echo ""; echo "export PATH=\$PATH:$WORKSPACE_PATH/.local/bin/quarto" ; echo "" ) >> ~/.profile
source ~/.profile
```
