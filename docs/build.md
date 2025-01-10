# Building the `FMBench` Python package

If you would like to build a dev version of `FMBench` for your own development and testing purposes, the following steps describe how to do that.

1. Clone the `FMBench` repo from GitHub.

1. Make any code changes as needed.

1. Install [`poetry`](https://pypi.org/project/poetry/).
   
    ```{.bash}
    pip install poetry mkdocs-material mknotebooks
    sudo apt-get install tree
    ```

1. Change directory to the `FMBench` repo directory and run poetry build.

    ```{.bash}
    poetry build
    ```

1. The `.whl` file is generated in the `dist` folder. Install the `.whl` in your current Python environment.

    ```{.bash}
    pip install dist/fmbench-X.Y.Z-py3-none-any.whl
    ```

1. Run `FMBench` as usual through the `FMBench` CLI command.

1. You may have added new config files as part of your work, to make sure these files are called out in the `manifest.txt` run the following command. This command will overwrite the existing `manifest.txt` and `manifest.md` files. Both these files need to be committed to the repo. Reach out to the maintainers of this repo so that they can add new or modified config files to the blogs bucket (the CloudFormation stack would fail if a new file is added to the manifest but is not available for download through the S3 bucket).

    ```{.bash}
    python create_manifest.py
    ```

1. To create updated documentation run the following command. You need to be added as a contributor to the `FMBench` repo to be able to publish to the website, so this command would not work for you if you are not added as a contributor to the repo.

    ```{.bash}
    mkdocs gh-deploy
    ```


