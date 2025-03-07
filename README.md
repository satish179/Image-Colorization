# Image-Colorization
# Image Colorization and Enhancement using DeOldify

## Project Overview

This project focuses on **image colorization and enhancement** using **DeOldify**, a state-of-the-art deep learning model based on Generative Adversarial Networks (GANs). The project takes grayscale images and colorizes them while also enhancing the details for a more visually appealing output.

## Features

- **Upload grayscale images for colorization**
- **Automatic image enhancement** to improve quality
- **Comparison display**: Side-by-side visualization of grayscale and colorized images
- **Uses DeOldify (Pretrained GAN model)**

##  Installation & Setup

### 🔹 Step 1: Clone the Repository

```sh
!git clone https://github.com/jantic/DeOldify.git
%cd DeOldify
```

### 🔹 Step 2: Install Required Libraries

```sh
!pip install -r requirements.txt
```

### 🔹 Step 3: Download Pretrained Model

```sh
!mkdir 'models'
!wget -O ./models/ColorizeArtistic_gen.pth https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth
```

## Usage Instructions

### Upload and Colorize an Image

```python
from deoldify.visualize import get_image_colorizer
colorizer = get_image_colorizer(artistic=True)
colorizer.plot_transformed_image('input.jpg', render_factor=35, display_render_factor=True)
```

### Enhance the Colorized Image

```python
import cv2

def enhance_image(image_path):
    image = cv2.imread(image_path)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    enhanced_img = cv2.merge((l_eq, a, b))
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2BGR)
    cv2.imwrite('enhanced.jpg', enhanced_img)
```

###  Display Comparison (Grayscale vs. Colorized)

```python
import matplotlib.pyplot as plt
import cv2

def compare_images(original, colorized):
    gray_img = cv2.imread(original, cv2.IMREAD_GRAYSCALE)
    color_img = cv2.imread(colorized)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(gray_img, cmap='gray')
    plt.title('Grayscale Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    plt.title('Colorized Image')
    plt.show()

compare_images('input.jpg', 'enhanced.jpg')
```

## 📂 File Structure

```
├── DeOldify
│   ├── models/               # Pretrained GAN model weights
│   ├── notebooks/            # Jupyter Notebooks for running the model
│   ├── images/               # Sample images
│   ├── deoldify/             # Core DeOldify files
│   ├── requirements.txt      # Required dependencies
│   ├── README.md             # Documentation
```

## Technologies Used

- **Python**
- **PyTorch**
- **OpenCV**
- **Matplotlib**
- **DeOldify (GAN-based model)**

##  Future Enhancements

- Implement real-time video colorization
- Develop a web-based interface using Flask or Streamlit

## Author & Credits

- **DeOldify Developers** for the original implementation
- **Satish Tirumala** for customization & enhancement

##  Acknowledgments

This project is built using **DeOldify** and **DeepAI’s pretrained models**.

## 📝 License

This project follows the **MIT License**. 

**Project Link:** https://colab.research.google.com/drive/1uesvHR2LfFpP5bmddxbhMSTYb7mCCkqL#scrollTo=A5VHAWPoFX84
