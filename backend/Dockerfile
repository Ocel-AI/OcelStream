FROM nvcr.io/nvidia/deepstream:7.1-gc-triton-devel

# Optional: install additional packages if needed
# RUN apt-get update && apt-get install -y vim git
# RUN ls bin/bash /opt/nvidia/deepstream/deepstream-7.1/sources/
WORKDIR /
COPY ./run.sh  /
RUN  /bin/bash /run.sh
WORKDIR /opt/nvidia/deepstream/deepstream-7.1/sources/
CMD [ "/bin/bash" ]