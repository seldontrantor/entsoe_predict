FROM ubuntu:latest
LABEL authors="amin"

ENTRYPOINT ["top", "-b"]