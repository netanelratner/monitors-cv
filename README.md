- `v1/detect_codes`
    Gets an image returns qr codes in the structure:
    ```json
    [{"data":"xxx","top":1,"left":1,"bottom":100,"right":100},]
    ```
- `v1/align_image`
    Gets and image and returns an image (jpeg)

- `v1/run_ocr`
    gets:
    ```json
    {
        "image": "base64 jpeg encoded image",
        "segments":[
        {
            "top":0,
            "left":0,
            "bottom":100,
            "right":100,
            "name":"monitor-blood",
        },]
    }
    ```
    returns:
    ```json
    [ {"segment_name":"string","value":"string"},]
    ```


## Install:

```bash
pip insatll -e .
```

## Test

```bash
pytest
```

## Run

```bash
cvmonitor
```

## Build docker:

```bash
docker build .
```

## Develop:

- Ubuntu 18.04 (16.04 & 19.10 may work)



```bash
# Install Dependancies:
sudo apt-get update && sudo apt-get install -yy wget libzbar0 libjpeg-turbo8-dev libz-dev python3-pip python3-venv git-lfs
# Create a virutal enviornment (once)
python3 -venv ~/envs/cvmonitors/
# Clone the repo:
git clone https://github.com/giladfr-rnd/monitors-cv && cd monitors-cv && git lfs pull
# Activate virtuale enviroment (every time)
source  ~/envs/cvmonitors/bin/activate
# Install in dev mode
pip install -e .
# Run tests
pytset
# maybe install matplotlib some packages for easier development
pip install matplotlib pdbpp 

## For openvino (intel inference engine)
sudo scripts/install-openvino.sh
scripts/install-openvino-python.sh
```


## Run Server:


### Install docker & docker compose:

Instructions from  docker.com:
```bash
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
sudo curl -L "https://github.com/docker/compose/releases/download/1.25.4/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### Start the server

From the folder of this repo run

```bash
git pull
docker-compose up
```

Stop the server:

- Ctrl + C

optionally:

```bash
docker-compose rm
```


- If you want to change parameters, edit them in the docker compose file.

## Simulator

Start the docker with 

```bash
cvmonitor/generator/generate.py --help
```

To see options.

Basically you need the send option which will generate devices and send images to the server, so just run


```bash
docker run <image-name> --net host cvmonitor/generator/generate.py --send --url <my-server-url>
```

And it will generate a new set of devices and send them to the server in url you set. 

Run 

```bash
docker run <image-name> --net host cvmonitor/generator/generate.py --delete-all --url <my-server-url>
```

To delete all devices from server