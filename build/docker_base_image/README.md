# Building The Data Cleanup Pipeline Docker Image
The Dockefile in this directory contains all the commands, in order, needed to build the **Data Cleanup Pipeline** docker image.


* Run the "make" command to build the **Data Cleanup Pipeline** docker image (output: docker image called "data_cleanup_pipeline" and a tag with today's date and time):
```
    make build_docker_image
```

* Login to docker hub. When prompted, enter your password and press enter:
```
    make login_to_dockerhub username=(enter your docker login here) email=(enter your email here)
```

* Upload your image to docker hub:
```
    make push_to_dockerhub
```
