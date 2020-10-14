# MLMicroserviceTemplate

## Overview

This is a template application for developing a machine learning model and serving the
 results of it on a webserver. This allows the model to work with a server such as [\[This One\]](https://github.com/UMass-Rescue/PhotoAnalysisServer)

The entire application is containerized with Docker, so the entire development and 
deployment process is portable.





## Initial Setup

To run this application, there are some commands that must be run. All of these should be
done via the command line in the root directly of the project folder.

Ensure that you have Docker running on your computer before attempting these commands.


### Configure Application

When creating a model with this template, there are two options that must be configured before
you can get full functionality from the template. These options are set once per model, and will
not need to be changed by you in the future during development.

#### [Configuration] Step 1
Open the file `.env` in the root directory. Change the first line from:
`NAME=example_model` to a more descriptive name for your model.

> [**Example**]
> If your model is able to detect soda cans, the first line of your `.env` file may look like the following:
> ```text
> NAME=soda_detect
> ```

#### [Configuration] Step 2 (Optional)
This step must only be done if you are running multiple template Docker containers on the same computer.

Open the file `.env` in the root directory. Change the first line from:
`PORT=5005` to another port that you have not already used.


### Build and Run Application

Once you have configured the template, you must run these commands to build and run the model.

#### [Build & Run] Step 0
If this is the first time you are setting up a model template, you must manually create a "volume".
This is used to track and share uploaded images.

**You only need to run this command once per computer, regardless of how many different models
 you will be running on the machine**

```cmd
docker volume create --name=photoanalysisserver_images
``` 


#### Step 1
Download dependencies with Docker and build container

```cmd
docker-compose build
```
#### Step 2
Start application
```cmd
docker-compose up -d
```


## Model Development

When working with models in this template, there are some useful commands and intricacies 
to keep in mind.




### [Dev Command] Connect to Application's Terminal
In the initial setup, you set the name of the application. Using that name as `$NAME`, you may run
the following command to connect your terminal window to that of the Docker container. This allows you to run commands such as
`python` to debug and test your model.
```cmd
docker exec -it $NAME bash
```
> [**Example**]
> If your model is named soda_detect, run the following command to connect to the terminal for the model
> ```cmd
> docker exec -it soda_detect bash
> ```


## Web Server Configuration

