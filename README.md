# Monitor reader computer vision service 

Web service for computer vision compoenents for the monitor-monitoring system.

This web service has a few endpoints, check /apidocs for thier description, in general:

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

