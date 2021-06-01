# TextGenerator

- This is a tools for ocr dataset, text detection, fonts classification dataset generate.
- This is the most convenient tool for generating ocr data, text detection data, and font recognition
## Realized functions:

-Generate text maps with different fonts, font sizes, colors, and rotation angles based on different corpora
-Support multi-process fast generation
-The text map is filled into the layout block according to the specified layout mode
-Find smooth areas in the image as layout blocks
-Support the extraction and export of blocks in the text area (export json file, txt file and picture file, can generate voc data, ICDAR_LSVT data set format!)
-Support annotations for each text level (stored in the json file of lsvt)
-Support users to configure various generation configurations (image reading, generation path, various probabilities)
## Effect preview

### Generate picture example:

![](img/pic_7f6cb78368edaf8347a8f0ce7e5a46c2df4f3ddd.jpg)
### Text map example:

![](img/fragment_6fc1b6ac180755dea3dfe711550251708b5e2ce519.jpg)

![](img/fragment_178b7da018e0d84c80b1455be4cc099bc68a07271.jpg)

![](img/fragment_ca71322eec0332fb3f6bb2a213c22f4a183c69da7.jpg)

![](img/fragment_f712bd7187d446b5fd5daf0ee0c6cb33ad26f98710.jpg)

### Rotating rectangle example

![](img/rotate_rect.png)

### Example of a single text bounding box

![](img/char_box.png)

-Environment installation (Python3.6+, conda environment is recommended)
        
    ```
    # step 1
    pip install requirements.txt
    # step 2
    sh make.sh
    ```
  
-Edit the configuration file `config.yml` (optional)
    
- Execute build script

    ```
    python3 run.py
    ```
  
-Generated data
    
    The generated data is stored in the directory specified by `provider> layout> out_put_dir` in `config.yml`.


