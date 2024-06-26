# Image Captioning AI

This project is an Image Captioning AI that combines computer vision and natural language processing to generate captions for images. It uses a pre-trained image recognition model (VGG16) to extract features from images and a recurrent neural network (RNN) or transformer-based model to generate captions.

## Features

- Upload an image and get a generated caption describing the image.
- Uses VGG16 for feature extraction.
- Utilizes a trained RNN model for caption generation.
- Simple and intuitive web interface.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/shahram8708/image-captioning-ai.git
    cd image-captioning-ai
    ```

2. **Create a virtual environment and activate it:**

    ```sh
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Ensure the following files are in the root directory:**
    - `caption_model.h5` (Pre-trained caption generation model)
    - `tokenizer.pickle` (Pre-trained tokenizer)

## Usage

1. **Run the Flask app:**

    ```sh
    python app.py
    ```

2. **Open your browser and navigate to:**

    ```
    http://127.0.0.1:5000/
    ```

3. **Upload an image and get the generated caption.**

## Folder Structure

```
project/
├── app.py
├── templates/
│   ├── index.html
├── static/
│   ├── styles.css
│   ├── script.js
├── uploads/
├── caption_model.h5
├── tokenizer.pickle
├── requirements.txt
└── README.md
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
