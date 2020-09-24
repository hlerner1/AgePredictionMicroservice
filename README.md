# MLMicroserviceTemplate

## Overview

This is a template application for developing a machine learning model and serving the results of it on a webserver.

The entire application is containerized with Docker, so development is easily portable.


## Getting Started

1. Build docker container
```cmd
docker-compose build
```
2. Start server and container
```cmd
docker-compose up -d
```

3. Connect to terminal for development
```cmd
docker exec -it server bash
```

## Web Server Configuration

Set the desired port of the web server in the `.env` file.