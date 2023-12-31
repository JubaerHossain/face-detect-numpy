# face-detect-numpy

## Install

```python
pip install flask opencv-python mtcnn mysql-connector-python pika tensorflow

```

### Install RabbitMQ

#### Install RabbitMQ: Download and install RabbitMQ from the official website (https://www.rabbitmq.com/download.html). Follow the installation instructions for Windows.

```bash
   rabbitmq-server

```

### Enable RabbitMQ Management Plugin (optional):

To use the RabbitMQ Management Console, which provides a web-based interface for managing and monitoring RabbitMQ, you need to enable the Management Plugin.
In the RabbitMQ Command Prompt, run the following command:
bash

```bash
rabbitmq-plugins enable rabbitmq_management
```

##### Access the RabbitMQ Management Console (optional):

<p>Open a web browser and navigate to http://localhost:15672/ to access the RabbitMQ Management Console.
Log in using the default credentials:<p>
```bash
Username: guest
Password: guest
```

## Run

```python
    python update.py
```
